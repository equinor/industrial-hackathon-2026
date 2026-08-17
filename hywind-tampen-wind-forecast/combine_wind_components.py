"""Convert WindFEELS U/V component measurements into combined wind speeds.

Example:
    python combine_wind_components.py --input val.parquet --output targets.csv
"""

from __future__ import annotations

import argparse
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
        raise ValueError(f"{path} must contain a Time column or have a DatetimeIndex.")

    if not table.index.is_unique:
        raise ValueError(f"{path} contains duplicate timestamps.")
    return table.sort_index()


def combine_wind_components(components: pd.DataFrame) -> pd.DataFrame:
    """Convert matching ``<asset>_U`` and ``<asset>_V`` columns into wind speeds."""
    combined = {}
    for column in components.columns:
        column_name = str(column)
        if not column_name.endswith("_U"):
            continue
        asset_name = column_name[:-2]
        v_column = f"{asset_name}_V"
        if v_column not in components.columns:
            raise ValueError(f"Missing V component '{v_column}' for '{column_name}'.")
        combined[asset_name] = np.sqrt(
            components[column].astype(float) ** 2 + components[v_column].astype(float) ** 2
        )

    if not combined:
        raise ValueError("Input must contain at least one matching <asset>_U/<asset>_V pair.")
    result = pd.DataFrame(combined, index=components.index)
    result.index.name = "Time"
    return result


def write_table(table: pd.DataFrame, path: Path) -> None:
    """Write a combined target table to CSV or Parquet with a Time column."""
    output = table.reset_index()
    if path.suffix.lower() == ".parquet":
        output.to_parquet(path, index=False)
    elif path.suffix.lower() in {".csv", ".txt"}:
        output.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use CSV or Parquet.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="U/V component data.")
    parser.add_argument("--output", type=Path, required=True, help="Combined wind-speed table.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    write_table(combine_wind_components(load_table(arguments.input)), arguments.output)


if __name__ == "__main__":
    main()