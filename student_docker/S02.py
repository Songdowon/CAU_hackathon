"""S02: controlled layer masks and training-only CE-gradient-ratio selection.

Train via: python tools/run_exp.py configs/S02.yaml
Probe only: python S02.py --config configs/S02.yaml --probe-only
The unchanged teammate training loop is reused with a process-local selector.
"""
import argparse
import contextlib
import hashlib
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


def layer_groups(model):
    groups = {f"block.{i}": list(block.parameters()) for i, block in enumerate(model.backbone.blocks)}
    groups["norm"] = list(model.backbone.norm.parameters())
    groups["head"] = list(model.backbone.head.parameters())
    return groups


def apply_selection(model, blocks, *, train_norm, train_head):
    count = len(model.backbone.blocks)
    if len(set(blocks)) != len(blocks) or any(type(i) is not int or i < 0 or i >= count for i in blocks):
        raise ValueError("Block indices must be unique integers in model range")
    if not blocks and not train_norm and not train_head:
        raise ValueError("Selection would have no trainable parameters")
    for p in model.parameters():
        p.requires_grad_(False)
    modules = [model.backbone.blocks[i] for i in blocks]
    if train_norm:
        modules.append(model.backbone.norm)
    if train_head:
        modules.append(model.backbone.head)
    for module in modules:
        for p in module.parameters():
            p.requires_grad_(True)
    # Frozen intervening blocks still participate in autograd.
    return [p for p in model.parameters() if p.requires_grad]


def summarize_norms(forget, retain, *, epsilon=1e-8, min_norm=1e-8):
    if not forget or len(forget) != len(retain) or epsilon <= 0 or min_norm < 0:
        raise ValueError("Invalid norm samples or ratio thresholds")
    if any(not math.isfinite(v) or v < 0 for v in [*forget, *retain]):
        raise ValueError("Gradient norms must be finite and nonnegative")
    f, r = statistics.mean(forget), statistics.mean(retain)
    return {"forget_mean_norm": f, "retain_mean_norm": r,
            "forget_std_norm": statistics.pstdev(forget), "retain_std_norm": statistics.pstdev(retain),
            "ratio": f / (r + epsilon), "eligible": f > min_norm and r > min_norm,
            "forget_batch_norms": forget, "retain_batch_norms": retain}


def choose_blocks(rows, count):
    if type(count) is not int or count < 1:
        raise ValueError("Block budget must be a positive integer")
    eligible = [r for r in rows if r["group"].startswith("block.") and r["eligible"] and math.isfinite(r["ratio"])]
    eligible.sort(key=lambda r: (-r["ratio"], int(r["group"].split(".")[1])))
    if len(eligible) < count:
        raise ValueError("Not enough blocks with a reliable nonzero gradient signal")
    return sorted(int(r["group"].split(".")[1]) for r in eligible[:count])


@contextlib.contextmanager
def probe_state(model, seed):
    modes = [(m, m.training) for m in model.modules()]
    flags = [(p, p.requires_grad) for p in model.parameters()]
    py_state, np_state = random.getstate(), np.random.get_state()
    device = next(model.parameters()).device
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            if device.type == 'cuda':
                torch.cuda.set_rng_state(torch.Generator(device=device).manual_seed(seed).get_state(), device)
            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            for params in layer_groups(model).values():
                for p in params:
                    p.requires_grad_(True)
            yield
        finally:
            for p, flag in flags:
                p.requires_grad_(flag)
            for module, mode in modes:
                module.training = mode
            random.setstate(py_state)
            np.random.set_state(np_state)


def probe_gradient_ratios(model, loader_factory, *, batches=8, seed=1702, epsilon=1e-8, min_norm=1e-8):
    if type(batches) is not int or batches < 2:
        raise ValueError("At least two probe batches are required")
    groups = layer_groups(model)
    parameters = [p for group in groups.values() for p in group]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("Layer groups must not share parameters")
    norms = {name: {"forget": [], "retain": []} for name in groups}
    device = next(model.parameters()).device
    sample_counts = []
    with probe_state(model, seed):
        loaders = loader_factory()
        iterators = {split: iter(loaders[split]) for split in ("forget", "retain")}
        for batch_index in range(batches):
            try:
                paired = {split: next(it) for split, it in iterators.items()}
            except StopIteration as error:
                raise ValueError("Probe requested more batches than the released split provides") from error
            size = min(len(paired["forget"][1]), len(paired["retain"][1]))
            if size < 2:
                raise ValueError("Probe batch is too small")
            sample_counts.append(size)
            for split in ("forget", "retain"):
                x, y = paired[split]
                # FP32 CE gradients avoid small-denominator AMP quantization effects.
                logits = model(x[:size].to(device))
                loss = F.cross_entropy(logits.float(), y[:size].to(device), reduction="mean")
                gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
                offset = 0
                for name, params in groups.items():
                    squared = torch.zeros((), device=device)
                    for grad in gradients[offset:offset + len(params)]:
                        if grad is not None:
                            squared += grad.detach().float().square().sum()
                    norms[name][split].append(math.sqrt(squared.item()))
                    offset += len(params)
                del gradients, logits, loss
            print(f"[S02 probe] {batch_index + 1}/{batches}", flush=True)
    return [{"group": name, "parameter_count": sum(p.numel() for p in params),
             "batch_sizes": sample_counts, "examples_per_split": sum(sample_counts),
             **summarize_norms(norms[name]["forget"], norms[name]["retain"], epsilon=epsilon, min_norm=min_norm)}
            for name, params in groups.items()]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def select_for_config(model, cfg, base):
    spec = cfg["selection"]
    mode = spec["mode"]
    rows = []
    probe_seconds = 0.0
    count = len(model.backbone.blocks)
    if mode == "gradient_ratio":
        probe = spec["probe"]
        def factory():
            return base.get_loaders(model, batch_size=probe["batch_size"], workers=0, seed=probe["seed"],
                                    split_pt=cfg["data"]["split"], forget_json=cfg["data"]["forget"])
        start = time.monotonic()
        rows = probe_gradient_ratios(model, factory, batches=probe["batches"], seed=probe["seed"],
                                     epsilon=probe["epsilon"], min_norm=probe["min_norm"])
        probe_seconds = time.monotonic() - start
        blocks = choose_blocks(rows, spec["count"])
    elif mode == "last":
        n = spec["count"]
        if type(n) is not int or not 1 <= n <= count:
            raise ValueError("Invalid number of last blocks")
        blocks = list(range(count - n, count))
    elif mode == "head_only":
        blocks = []
        if spec["train_norm"] or not spec["train_head"]:
            raise ValueError("head_only means classifier only, with final norm frozen")
    elif mode == "all":
        blocks = list(range(count))
    else:
        raise ValueError(f"Unknown selection mode: {mode}")
    if mode == "all":
        for p in model.parameters():
            p.requires_grad_(True)
        params = list(model.parameters())
    else:
        params = apply_selection(model, blocks, train_norm=spec["train_norm"], train_head=spec["train_head"])
    metadata = {"mode": mode, "block_indices_zero_based": blocks, "block_numbers_one_based": [i + 1 for i in blocks],
                "trainable_names": [name for name,p in model.named_parameters() if p.requires_grad],
                "trainable_parameter_count": sum(p.numel() for p in params),
                "total_parameter_count": sum(p.numel() for p in model.parameters()),
                "probe_seconds": probe_seconds, "gradient_statistics": rows}
    return params, metadata


def write_metadata(path, data, *, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if exclusive:
        with path.open('x') as stream:
            stream.write(encoded)
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/S02.yaml")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg["model"]["num_classes"] != 100 or cfg["model"]["arch"] != "vit_base_patch16_224.mae":
        raise ValueError("Competition architecture must remain unchanged")
    checkpoint = Path(cfg["output"]["save_path"])
    report = Path(cfg["output"]["selection_report"])
    if args.probe_only:
        report = report.with_name(report.stem + ".probe.json")
    import S02_reference as base
    metadata = {"experiment": Path(args.config).stem, "config": cfg, "status": "preparing",
                "probe_objective": "mean true-label CE, separately on released forget and retain",
                "ratio_definition": "mean batch forget L2 norm / (mean batch retain L2 norm + epsilon)",
                "scores_are_not_unlearning_guarantees": True,
                "source_sha256": {str(path): sha256(path) for path in (Path(__file__), Path(base.__file__), Path(args.config), Path(cfg['model']['mo_ckpt']), Path(cfg['data']['split']), Path(cfg['data']['forget']), Path('utils/data.py'), Path('imagenet_vit.py'))},
                'software': {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__}}
    if args.probe_only:
        import fcntl
        with open("/tmp/hackathon_gpu.lock", "a") as lock:
            print("[S02] Waiting for shared GPU lock (probe only)", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            if report.exists():
                raise FileExistsError(f"Preserve existing probe report: {report}")
            base.set_seed(cfg["seed"])
            model = base.load_mo(cfg["model"]["mo_ckpt"], 100, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            _, selected = select_for_config(model, cfg, base)
            metadata.update(selected)
            metadata["status"] = "probe_completed_no_training"
            write_metadata(report, metadata, exclusive=True)
            print(json.dumps({"report": str(report), "selected_blocks": selected["block_numbers_one_based"]}), flush=True)
        return
    if checkpoint.exists() or report.exists():
        raise FileExistsError("Output checkpoint or selection report already exists; use a new run ID")
    write_metadata(report, metadata, exclusive=True)
    original_selector, original_argv = base.set_trainable, sys.argv[:]
    def selector(model, _legacy_count):
        params, selected = select_for_config(model, cfg, base)
        metadata.update(selected)
        metadata["status"] = "selected_training_pending"
        write_metadata(report, metadata)
        print(f"[S02] selected blocks (1-based): {metadata['block_numbers_one_based']}; trainable={metadata['trainable_parameter_count']:,}", flush=True)
        return params
    try:
        base.set_trainable = selector
        sys.argv = [str(Path(base.__file__)), "--config", args.config]
        base.main()
        if not checkpoint.is_file():
            raise RuntimeError("Training returned without creating its checkpoint")
        metadata["status"] = "training_completed_evaluation_pending"
        write_metadata(report, metadata)
    except BaseException as error:
        metadata.update(status="failed", error=type(error).__name__ + ": " + str(error))
        write_metadata(report, metadata)
        raise
    finally:
        base.set_trainable = original_selector
        sys.argv = original_argv


if __name__ == "__main__":
    main()
