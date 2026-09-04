#!/usr/bin/env python3
"""Score one experiment checkpoint on the local public validation set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


WORKSPACE = Path(os.environ.get("STUDENT_WORKSPACE_ROOT", "/workspace")).resolve()
DATASET = Path(os.environ.get("DATASET_ROOT", str(WORKSPACE / "datasets"))).resolve()
SCORER = Path(os.environ.get("TRUSTED_SCORER_ROOT", "/opt/hackathon/scorer"))


def _checkpoint(raw: str) -> Path:
    candidate = Path(raw)
    path = (WORKSPACE / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if WORKSPACE not in path.parents or path.suffix != ".pt":
        raise ValueError("checkpoint must be a .pt file below /workspace")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"checkpoint is missing or not a regular file: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="for example models/experiment-007.pt")
    parser.add_argument("--device", default=os.environ.get("SCORING_DEVICE", "cuda"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("SCORING_BATCH_SIZE", "128")),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SCORING_WORKERS", "4")),
    )
    args = parser.parse_args()

    try:
        checkpoint = _checkpoint(args.checkpoint)
    except ValueError as exc:
        parser.error(str(exc))
    results = WORKSPACE / "results"
    results.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = results / f"{checkpoint.stem}-validation-{timestamp}.json"

    with tempfile.TemporaryDirectory(prefix="local-validation-", dir=results) as raw_tmp:
        converted = Path(raw_tmp) / "model.safetensors"
        subprocess.run(
            [
                sys.executable,
                str(SCORER / "convert_checkpoint.py"),
                "--input",
                str(checkpoint),
                "--output",
                str(converted),
            ],
            check=True,
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(SCORER / "score_unlearning.py"),
                "score",
                "--phase",
                "validation",
                "--split",
                str(DATASET / "splits/student_split.pt"),
                "--refs",
                str(DATASET / "validation_cache/refs.pt"),
                "--image-root",
                str(DATASET / "imagenet_released"),
                "--ckpt",
                str(converted),
                "--mo-cache",
                str(DATASET / "validation_cache/M_o__validation.npz"),
                "--tag",
                checkpoint.name,
                "--report",
                str(report),
                "--device",
                args.device,
                "--batch-size",
                str(args.batch_size),
                "--workers",
                str(args.workers),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"scorer did not produce a valid report: {exc}") from exc

    print(f"model       : {checkpoint.relative_to(WORKSPACE)}")
    print(f"AUS         : {payload['AUS']:.8f}")
    print(f"RUS_o       : {payload['RUS_o']:.8f}")
    print(f"final_score : {payload['final_score']:.8f}")
    print(f"report      : {report.relative_to(WORKSPACE)}")
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
