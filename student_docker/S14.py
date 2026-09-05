#!/usr/bin/env python3
"""S14: private-robust weighted soup built entirely on CPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch
import yaml

DEFAULT_HEAD_PREFIXES = ("head.", "backbone.head.")


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no model state: {path}")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError(f"checkpoint contains non-tensor state: {path}")
    return state


def is_head(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(p) or f".{p}" in name for p in prefixes)


def validate(primary: Mapping[str, torch.Tensor], secondary: Mapping[str, torch.Tensor]) -> None:
    if primary.keys() != secondary.keys():
        missing = sorted(set(primary) - set(secondary))
        extra = sorted(set(secondary) - set(primary))
        raise ValueError(f"state keys differ: missing={missing[:5]} extra={extra[:5]}")
    for name, tensor in primary.items():
        other = secondary[name]
        if tensor.shape != other.shape:
            raise ValueError(f"shape mismatch for {name}: {tensor.shape} vs {other.shape}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text())
    inputs = cfg["inputs"]
    if len(inputs) != 2:
        raise ValueError("S14 expects exactly two input checkpoints")
    weights = [float(x["weight"]) for x in inputs]
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    weights = [w / total for w in weights]
    paths = [Path(x["path"]) for x in inputs]
    states = [load_state(p) for p in paths]
    primary_index = int(cfg.get("head_source_index", 0))
    if primary_index not in (0, 1):
        raise ValueError("head_source_index must be 0 or 1")
    validate(states[0], states[1])
    prefixes = tuple(cfg.get("head_prefixes", DEFAULT_HEAD_PREFIXES))

    merged: dict[str, torch.Tensor] = {}
    head_keys: list[str] = []
    averaged = 0
    copied_nonfloat = 0
    max_abs_to_primary = 0.0
    for name, a in states[0].items():
        b = states[1][name]
        primary = states[primary_index][name]
        if is_head(name, prefixes):
            out = primary.detach().cpu().clone()
            head_keys.append(name)
        elif not (a.is_floating_point() or a.is_complex()):
            out = primary.detach().cpu().clone()
            copied_nonfloat += 1
        else:
            out = (a.detach().cpu().float() * weights[0] + b.detach().cpu().float() * weights[1]).to(a.dtype)
            averaged += 1
        if out.is_floating_point() and not torch.isfinite(out).all():
            raise ValueError(f"non-finite output tensor: {name}")
        if out.is_floating_point():
            delta = (out.float() - primary.detach().cpu().float()).abs().max().item()
            max_abs_to_primary = max(max_abs_to_primary, delta)
        merged[name] = out

    output = Path(cfg["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": merged}, output)

    # Reload once on CPU so the saved artifact, not only the in-memory state, is verified.
    saved = load_state(output)
    validate(merged, saved)
    for name in merged:
        if not torch.equal(merged[name], saved[name]):
            raise ValueError(f"saved tensor mismatch: {name}")
    for name in head_keys:
        if not torch.equal(saved[name], states[primary_index][name]):
            raise ValueError(f"head was not preserved exactly: {name}")

    audit = {
        "experiment": "S14",
        "method": "private-robust weighted soup; average non-head floating tensors and preserve best-private head",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "inputs": [
            {"path": str(p), "normalized_weight": w, "local": inputs[i].get("local"), "private": inputs[i].get("private")}
            for i, (p, w) in enumerate(zip(paths, weights))
        ],
        "head_source": str(paths[primary_index]),
        "head_keys": head_keys,
        "summary": {
            "total_tensors": len(merged),
            "averaged_float_tensors": averaged,
            "copied_head_tensors": len(head_keys),
            "copied_nonfloat_tensors": copied_nonfloat,
            "max_abs_delta_from_head_source": max_abs_to_primary,
        },
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "sha256": sha256(output),
        "queue_registered": False,
        "gpu_used": False,
    }
    audit_path = Path(cfg["audit"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + chr(10))
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/S14.yaml")
    args = parser.parse_args()
    audit = build(Path(args.config))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
