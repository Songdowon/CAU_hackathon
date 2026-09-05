#!/usr/bin/env python3
"""Estimate private leaderboard score from a local validation JSON.

V1 is deliberately small: local final plus a one-feature RUS optimism-gap
calibration fitted to the five documented submissions. It is a first-pass
screen, not a replacement for the remaining leaderboard submissions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_CALIBRATION = Path(__file__).resolve().parents[1] / "results" / "private_proxy_v1_calibration.json"


@dataclass(frozen=True)
class CalibrationPoint:
    name: str
    family: str
    local_final: float
    private: float
    rus_o: float


@dataclass(frozen=True)
class CalibrationModel:
    center: float
    intercept: float
    slope: float
    rus_min: float
    rus_max: float


def fit_rus_gap_calibration(points: Sequence[CalibrationPoint]) -> CalibrationModel:
    if len(points) < 2:
        raise ValueError("at least two calibration points are required")
    xs = [p.rus_o for p in points]
    gaps = [p.private - p.local_final for p in points]
    center = sum(xs) / len(xs)
    intercept = sum(gaps) / len(gaps)
    denom = sum((x - center) ** 2 for x in xs)
    if denom == 0:
        raise ValueError("RUS_o has no variation in calibration data")
    slope = sum((x - center) * (gap - intercept) for x, gap in zip(xs, gaps)) / denom
    return CalibrationModel(center, intercept, slope, min(xs), max(xs))


def estimate_private(
    local_final: float,
    rus_o: float,
    cka_f_o: float,
    model: CalibrationModel,
    guardrail: float = 0.03,
) -> dict:
    clipped_rus = min(max(rus_o, model.rus_min), model.rus_max)
    proxy = local_final + model.intercept + model.slope * (clipped_rus - model.center)
    guardrail_pass = cka_f_o <= guardrail
    return {
        "local_final": local_final,
        "RUS_o": rus_o,
        "CKA_f_o": cka_f_o,
        "private_proxy_v1": proxy,
        "selection_score": proxy if guardrail_pass else None,
        "guardrail_pass": guardrail_pass,
        "guardrail_CKA_f_o_max": guardrail,
        "rus_extrapolated": not (model.rus_min <= rus_o <= model.rus_max),
        "rus_used": clipped_rus,
    }


def load_validation_metrics(path: Path) -> tuple[float, float, float]:
    data = json.loads(path.read_text())
    try:
        local_final = float(data["final_score"])
        representation = data.get("representation_metric", {})
        rus_o = float(data["RUS_o"] if "RUS_o" in data else representation["RUS_o"])
        cka_f_o = float(data["CKA_f_o"] if "CKA_f_o" in data else representation["CKA_f_o"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unsupported validation JSON schema: {path}") from exc
    return local_final, rus_o, cka_f_o


def load_calibration(path: Path) -> tuple[list[CalibrationPoint], float]:
    data = json.loads(path.read_text())
    points = [
        CalibrationPoint(
            name=row["name"],
            family=row["family"],
            local_final=float(row["local_final"]),
            private=float(row["private"]),
            rus_o=float(row["RUS_o"]),
        )
        for row in data["points"]
    ]
    return points, float(data.get("guardrail_cka_f_o", 0.03))


def grouped_cv_stats(points: Sequence[CalibrationPoint]) -> dict:
    errors = []
    for family in sorted({p.family for p in points}):
        train = [p for p in points if p.family != family]
        test = [p for p in points if p.family == family]
        model = fit_rus_gap_calibration(train)
        for point in test:
            clipped = min(max(point.rus_o, model.rus_min), model.rus_max)
            pred = point.local_final + model.intercept + model.slope * (clipped - model.center)
            errors.append(pred - point.private)
    return {
        "scheme": "leave-one-family-out",
        "n_points": len(points),
        "n_families": len({p.family for p in points}),
        "mae": sum(abs(e) for e in errors) / len(errors),
        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "max_abs_error": max(abs(e) for e in errors),
    }


def score_paths(paths: Iterable[Path], calibration_path: Path) -> list[dict]:
    points, guardrail = load_calibration(calibration_path)
    model = fit_rus_gap_calibration(points)
    audit = grouped_cv_stats(points)
    outputs = []
    for path in paths:
        local_final, rus_o, cka_f_o = load_validation_metrics(path)
        result = estimate_private(local_final, rus_o, cka_f_o, model, guardrail)
        err = audit["max_abs_error"]
        result.update({
            "source": str(path),
            "empirical_error_band": [max(0.0, result["private_proxy_v1"] - err), min(1.0, result["private_proxy_v1"] + err)],
            "calibration": asdict(model),
            "audit": audit,
            "status": "exploratory_v1",
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
