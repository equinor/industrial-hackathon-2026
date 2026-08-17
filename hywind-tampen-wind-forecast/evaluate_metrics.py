"""Evaluate WindFEELS test predictions.

Examples:
    python evaluate_metrics.py --predictions submission.csv --targets targets.csv
    python evaluate_metrics.py --predictions submission.parquet --targets targets.parquet \
        --output metrics.json

Prediction columns must use one of these forms:
    HYT-HY09                   Combined wind-speed forecast (treated as q=0.5)
    HYT-HY09_q0.05             Combined wind-speed quantile forecast

The target file must contain combined wind-speed columns matching the submitted forecast
columns and a ``Time`` column or datetime index. Use ``combine_wind_components.py`` to
create combined targets from U/V component data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TIME_COLUMN_NAMES = {"time", "timestamp", "datetime", "date"}


def load_table(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet table and normalize its datetime index."""
    if path.suffix.lower() == ".parquet":
        table = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        table = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use CSV or Parquet.")

    time_column = next(
        (column for column in table.columns if str(column).strip().lower() in TIME_COLUMN_NAMES),
        None,
    )
    if time_column is not None:
        table[time_column] = pd.to_datetime(table[time_column], utc=True)
        table = table.set_index(time_column)
    elif isinstance(table.index, pd.DatetimeIndex):
        table.index = pd.to_datetime(table.index, utc=True)
    else:
        raise ValueError(
            f"{path} must contain a Time column or have a DatetimeIndex."
        )

    if not table.index.is_unique:
        raise ValueError(f"{path} contains duplicate timestamps.")
    return table.sort_index()


def parse_prediction_column(column: str) -> tuple[str, float]:
    """Parse a stable ``target`` or ``target_q<quantile>`` column name."""
    if "_q" not in column:
        return column, 0.5

    target_name, quantile_text = column.rsplit("_q", 1)
    if not target_name or not quantile_text:
        raise ValueError(
            f"Invalid prediction column '{column}'. Use '<target>' or '<target>_q<quantile>'."
        )

    try:
        quantile = float(quantile_text)
    except ValueError as error:
        raise ValueError(
            f"Invalid quantile in prediction column '{column}'. Use a value between 0 and 1."
        ) from error

    if not 0 < quantile <= 1:
        raise ValueError(
            f"Invalid quantile in prediction column '{column}'. Use a value between 0 and 1."
        )
    return target_name, quantile


def pinball_loss(actual: np.ndarray, predicted: np.ndarray, quantile: float) -> float:
    """Calculate mean pinball loss, matching sklearn's convention."""
    residual = actual - predicted
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1) * residual)))


def score_predictions(predictions: pd.DataFrame, targets: pd.DataFrame) -> dict:
    """Calculate per-target and overall metrics for all submitted forecasts."""
    common_timestamps = predictions.index.intersection(targets.index)
    if common_timestamps.empty:
        raise ValueError("Predictions and targets have no timestamps in common.")

    predictions = predictions.loc[common_timestamps]
    targets = targets.loc[common_timestamps]
    scores: dict[str, dict] = {}
    all_squared_errors: list[np.ndarray] = []
    all_pinball_losses: list[float] = []

    for column in predictions.columns:
        target_name, quantile = parse_prediction_column(str(column))
        if target_name not in targets.columns:
            raise ValueError(
                f"Prediction column '{column}' refers to missing target '{target_name}'."
            )

        paired_values = pd.concat(
            [targets[target_name].rename("actual"), predictions[column].rename("predicted")],
            axis=1,
        ).dropna()
        if paired_values.empty:
            raise ValueError(f"No non-null values to score for prediction column '{column}'.")

        actual = paired_values["actual"].to_numpy(dtype=float)
        predicted = paired_values["predicted"].to_numpy(dtype=float)
        losses = (actual - predicted) ** 2
        metrics = {
            "quantile": quantile,
            "n": int(len(paired_values)),
            "pinball_loss": pinball_loss(actual, predicted, quantile),
        }
        if quantile == 0.5:
            metrics["RMSE"] = float(np.sqrt(np.mean(losses)))
            all_squared_errors.append(losses)
        else:
            metrics["over_estimate_pct"] = float(np.mean(predicted > actual) * 100)
            metrics["mean_underestimation"] = float(np.mean(actual - predicted))

        scores[str(column)] = round_metrics(metrics)
        all_pinball_losses.append(metrics["pinball_loss"])

    overall = {"n_prediction_columns": len(scores)}
    if all_pinball_losses:
        overall["pinball_loss"] = float(np.mean(all_pinball_losses))
    if all_squared_errors:
        overall["RMSE"] = float(np.sqrt(np.mean(np.concatenate(all_squared_errors))))
    quantile_scores = [score for score in scores.values() if score["quantile"] != 0.5]
    if quantile_scores:
        overall["over_estimate_pct"] = float(
            np.mean([score["over_estimate_pct"] for score in quantile_scores])
        )
        overall["mean_underestimation"] = float(
            np.mean([score["mean_underestimation"] for score in quantile_scores])
        )
    scores["overall"] = round_metrics(overall)
    return scores


def round_metrics(metrics: dict) -> dict:
    """Round numeric output while retaining integer counts."""
    rounded = {}
    for name, value in metrics.items():
        rounded[name] = int(value) if name in {"n", "n_prediction_columns"} else (
            round(float(value), 6) if isinstance(value, (float, np.floating)) else value
        )
    return rounded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Submitted forecasts.")
    parser.add_argument("--targets", type=Path, required=True, help="Combined ground-truth wind speeds.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    scores = score_predictions(
        load_table(arguments.predictions),
        load_table(arguments.targets),
    )
    rendered_scores = json.dumps(scores, indent=2)
    print(rendered_scores)
    if arguments.output is not None:
        arguments.output.write_text(rendered_scores + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()