#!/usr/bin/env python3
"""Private Proxy V2 fitted only to the four most recent documented submissions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from tools.private_proxy_score import CalibrationPoint, estimate_private, fit_rus_gap_calibration, load_validation_metrics
except ModuleNotFoundError:
    from private_proxy_score import CalibrationPoint, estimate_private, fit_rus_gap_calibration, load_validation_metrics


DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "results" / "private_proxy_v2_recent_calibration.json"


def _point(row: dict) -> CalibrationPoint:
    return CalibrationPoint(row["name"], row["family"], float(row["local_final"]), float(row["private"]), float(row["RUS_o"]))


def load_recent_calibration(path: Path = DEFAULT_CALIBRATION) -> tuple[list[CalibrationPoint], list[CalibrationPoint], float, float]:
    data = json.loads(path.read_text())
    points = [_point(row) for row in data["points"]]
    holdouts = [_point(row) for row in data.get("temporal_holdouts", [])]
    return points, holdouts, float(data.get("guardrail_cka_f_o", 0.03)), float(data["conservative_max_abs_error"])


def temporal_holdout_audit(points: Sequence[CalibrationPoint], holdouts: Sequence[CalibrationPoint]) -> dict:
    if not holdouts:
        raise ValueError("at least one temporal holdout is required")
    model = fit_rus_gap_calibration(points)
    predictions = []
    errors = []
    for point in holdouts:
        used = min(max(point.rus_o, model.rus_min), model.rus_max)
        predicted = point.local_final + model.intercept + model.slope * (used - model.center)
        error = predicted - point.private
        errors.append(error)
        predictions.append({"name": point.name, "predicted": predicted, "private": point.private, "error": error})
    return {
        "scheme": "older-submission temporal holdout",
        "n_holdouts": len(holdouts),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "max_abs_error": max(abs(e) for e in errors),
        "predictions": predictions,
    }


def score_paths(paths: Sequence[Path], calibration_path: Path = DEFAULT_CALIBRATION) -> list[dict]:
    points, holdouts, guardrail, conservative_error = load_recent_calibration(calibration_path)
    model = fit_rus_gap_calibration(points)
    temporal = temporal_holdout_audit(points, holdouts)
    outputs = []
    for path in paths:
        local_final, rus_o, cka_f_o = load_validation_metrics(path)
        result = estimate_private(local_final, rus_o, cka_f_o, model, guardrail)
        proxy = result.pop("private_proxy_v1")
        result.update({
            "metric_name": "private_proxy_v2_recent",
            "private_proxy_v2_recent": proxy,
            "source": str(path),
            "formula": {"center": model.center, "intercept": model.intercept, "slope": model.slope, "rus_min": model.rus_min, "rus_max": model.rus_max},
            "temporal_holdout_audit": temporal,
            "conservative_error_band": [max(0.0, proxy - conservative_error), min(1.0, proxy + conservative_error)],
            "status": "exploratory_v2_recent",
        })
        outputs.append(result)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_json", nargs="+", type=Path)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args()
    outputs = score_paths(args.validation_json, args.calibration)
    payload = outputs[0] if len(outputs) == 1 else outputs
    print(json.dumps(payload, ensure_ascii=False, indent=None if args.compact else 2, sort_keys=True))


if __name__ == "__main__":
    main()
