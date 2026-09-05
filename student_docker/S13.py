#!/usr/bin/env python3
"""S13: anchor-preserving consensus merge for CPU-only checkpoint creation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
import yaml


DEFAULT_HEAD_PREFIXES = ("head.", "backbone.head.")


def _unwrap(payload: object, path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a mapping: {path}")
    state = payload.get("model", payload)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no model state: {path}")
    if not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError(f"checkpoint contains non-tensor model entries: {path}")
    return state


def load_state(path: Path) -> dict[str, torch.Tensor]:
    return _unwrap(torch.load(path, map_location="cpu", weights_only=True), path)


def _is_head(name: str, prefixes: Sequence[str]) -> bool:
    return any(name.startswith(p) or f".{p}" in name for p in prefixes)


def _validate_states(base: Mapping[str, torch.Tensor], anchor: Mapping[str, torch.Tensor], helpers: Sequence[Mapping[str, torch.Tensor]]) -> None:
    expected = set(base)
    for label, state in [("anchor", anchor), *[(f"helper[{i}]", s) for i, s in enumerate(helpers)]]:
        if set(state) != expected:
            raise ValueError(f"{label} keys do not match base keys")
        for name, tensor in state.items():
            if tensor.shape != base[name].shape:
                raise ValueError(f"{label} shape mismatch for {name}")
            if tensor.dtype != base[name].dtype:
                raise ValueError(f"{label} dtype mismatch for {name}")


def _consensus_delta(base: torch.Tensor, anchor: torch.Tensor, helpers: Sequence[torch.Tensor]) -> tuple[torch.Tensor, dict]:
    base32 = base.float()
    anchor_delta = anchor.float() - base32
    anchor_norm = float(anchor_delta.norm().item())
    active = anchor_delta != 0
    active_count = int(active.sum().item())
    helper_deltas = [h.float() - base32 for h in helpers]

    if anchor_norm == 0.0 or active_count == 0:
        return anchor_delta, {"active": active_count, "anchor_norm": anchor_norm, "merged_norm": anchor_norm, "cosine_to_anchor": 1.0, "helper_sign_agreement": [1.0 for _ in helpers]}

    anchor_sign = torch.sign(anchor_delta)
    magnitudes = [anchor_delta.abs()]
    agreements = []
    agreement_rates = []
    for delta in helper_deltas:
        agree = active & (torch.sign(delta) == anchor_sign) & (delta != 0)
        agreements.append(agree)
        agreement_rates.append(float(agree.sum().item() / active_count))
        magnitudes.append(torch.where(agree, delta.abs(), torch.full_like(delta, float("inf"))))

    ordered = torch.sort(torch.stack(magnitudes), dim=0).values
    counts = torch.ones_like(anchor_delta, dtype=torch.long)
    for agree in agreements:
        counts = counts + agree.long()
    low = ((counts - 1) // 2).unsqueeze(0)
    high = (counts // 2).unsqueeze(0)
    median_magnitude = 0.5 * (torch.gather(ordered, 0, low).squeeze(0) + torch.gather(ordered, 0, high).squeeze(0))
    candidate = torch.where(active, anchor_sign * median_magnitude, torch.zeros_like(anchor_delta))
    candidate_norm = candidate.norm()
    if not torch.isfinite(candidate_norm) or candidate_norm.item() == 0.0:
        candidate = anchor_delta
    else:
        candidate = candidate * (anchor_norm / candidate_norm)

    merged_norm = float(candidate.norm().item())
    cosine = float(torch.dot(candidate.reshape(-1), anchor_delta.reshape(-1)).item() / max(merged_norm * anchor_norm, 1e-30))
    return candidate, {"active": active_count, "anchor_norm": anchor_norm, "merged_norm": merged_norm, "cosine_to_anchor": cosine, "helper_sign_agreement": agreement_rates}


def merge_state_dict(base: Mapping[str, torch.Tensor], anchor: Mapping[str, torch.Tensor], helpers: Sequence[Mapping[str, torch.Tensor]], head_prefixes: Sequence[str] = DEFAULT_HEAD_PREFIXES, numerical_delta_max: float = 0.0) -> tuple[dict[str, torch.Tensor], dict]:
    if not helpers:
        raise ValueError("at least one helper checkpoint is required")
    _validate_states(base, anchor, helpers)
    merged: dict[str, torch.Tensor] = {}
    tensor_audit: dict[str, dict] = {}
    summary = {"total_tensors": len(base), "changed_float_tensors": 0, "head_tensors_copied": 0, "frozen_tensors_copied": 0, "numerical_frozen_tensors_copied": 0, "non_float_tensors_copied": 0}

    for name in base:
        b, a = base[name], anchor[name]
        if _is_head(name, head_prefixes):
            merged[name] = a.detach().cpu().clone()
            summary["head_tensors_copied"] += 1
            tensor_audit[name] = {"action": "copy_anchor_head"}
            continue
        if not (a.is_floating_point() or a.is_complex()):
            merged[name] = a.detach().cpu().clone()
            summary["non_float_tensors_copied"] += 1
            tensor_audit[name] = {"action": "copy_anchor_non_float"}
            continue
        if torch.equal(a, b):
            merged[name] = a.detach().cpu().clone()
            summary["frozen_tensors_copied"] += 1
            tensor_audit[name] = {"action": "copy_anchor_frozen"}
            continue
        if float((a.float() - b.float()).abs().max().item()) <= numerical_delta_max:
            merged[name] = b.detach().cpu().clone()
            summary["numerical_frozen_tensors_copied"] += 1
            tensor_audit[name] = {"action": "restore_base_numerical_drift", "max_abs_delta": float((a.float() - b.float()).abs().max().item())}
            continue

        delta, stats = _consensus_delta(b, a, [h[name] for h in helpers])
        out = (b.float() + delta).to(dtype=a.dtype, device="cpu")
        if not torch.isfinite(out).all():
            raise ValueError(f"non-finite output tensor: {name}")
        merged[name] = out
        summary["changed_float_tensors"] += 1
        tensor_audit[name] = {"action": "anchor_consensus", **stats}

    return merged, {"summary": summary, "tensors": tensor_audit}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_from_config(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    base_path = Path(cfg["base"])
    anchor_path = Path(cfg["anchor"])
    helper_paths = [Path(p) for p in cfg["helpers"]]
    output_path = Path(cfg["output"])
    audit_path = Path(cfg["audit"])
    prefixes = tuple(cfg.get("head_prefixes", DEFAULT_HEAD_PREFIXES))
    numerical_delta_max = float(cfg.get("numerical_delta_max", 0.0))

    base = load_state(base_path)
    anchor = load_state(anchor_path)
    helpers = [load_state(p) for p in helper_paths]
    merged, merge_audit = merge_state_dict(base, anchor, helpers, prefixes, numerical_delta_max)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged}, output_path)
    reloaded = load_state(output_path)
    if set(reloaded) != set(merged) or any(not torch.equal(reloaded[k], merged[k]) for k in merged):
        raise RuntimeError("saved checkpoint failed exact CPU reload verification")

    audit = {
        "schema_version": 1,
        "experiment": cfg.get("experiment", "S13"),
        "method": "anchor_preserving_consensus",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gpu_used": False,
        "numerical_delta_max": numerical_delta_max,
        "sources": {
            "base": str(base_path),
            "anchor": str(anchor_path),
            "helpers": [str(p) for p in helper_paths],
            "sha256": {str(p): _sha256(p) for p in [base_path, anchor_path, *helper_paths]},
        },
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        **merge_audit,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/S13.yaml"))
    args = parser.parse_args()
    audit = run_from_config(args.config)
    print(json.dumps({"output": audit["output"], "output_sha256": audit["output_sha256"], "summary": audit["summary"], "gpu_used": False}, indent=2))


if __name__ == "__main__":
    main()
