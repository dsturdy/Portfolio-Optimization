# case_io.py
"""
Helpers for reading the First Eagle case Excel file.

Later passed into optimize_portfolio.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


# ---- Excel layout assumptions -------------------------------------
# Current-portfolio table header is on excel row 27 (0-index = 26)
CURRENT_PORTFOLIO_HEADER_ROW = 26

def _percent_to_float(x) -> float:
    """
    Converts percents to decimals and handles NaN values.
    """

    # Already numeric (Excel % often stored as decimal, e.g. 0.29)
    if isinstance(x, (int, float)):
        if pd.isna(x):
            return 0.0
        v = float(x)
        # if > 1, assume it's a whole percent (e.g. 29 → 29%)
        return v / 100.0 if v > 1.0 else v

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return 0.0

    has_percent = "%" in s
    s = s.replace("%", "").strip()
    if s == "":
        return 0.0

    val = float(s)

    # If it had a '%' or is > 1, treat as whole percent
    if has_percent or val > 1.0:
        return val / 100.0
    else:
        # Already decimal (e.g. "0.4")
        return val


def _band_to_min_max(x) -> tuple[float, float]:
    """
    Parses allocation bands from the 'Min - Max %' column.
    """
    s = str(x).strip()

    # Blank or NaN → NO explicit constraint
    if s == "" or s.lower() == "nan":
        return float("nan"), float("nan")

    # Expects ranges like "10% - 40%"
    if "-" in s:
        lo_str, hi_str = s.split("-", 1)
        lo = _percent_to_float(lo_str)
        hi = _percent_to_float(hi_str)
        return lo, hi

    # Single number like "40%" → means 0–40%
    hi = _percent_to_float(s)
    return 0.0, hi


def load_current_portfolio_from_case(case_path: Path) -> pd.DataFrame:
    """
    Load the Current Portfolio block from the case Excel file.

    Returns DataFrame with columns:
        Ticker, AssetGroup, MinPct, MaxPct, Weight
    where all percentages are in decimal format
    """

    # Read the sheet with a FIXED header row
    df = pd.read_excel(
        case_path,
        sheet_name=0,
        header=CURRENT_PORTFOLIO_HEADER_ROW,
    )

    # Standardize column names (strip spaces)
    df.columns = [str(c).strip() for c in df.columns]

    # Required columns
    if "Ticker" not in df.columns:
        raise ValueError("Expected a 'Ticker' column in the current-portfolio table.")
    if "Current Allocations" not in df.columns:
        raise ValueError("Expected a 'Current Allocations' column in the current-portfolio table.")
    if "Asset Group" not in df.columns:
        raise ValueError("Expected an 'Asset Group' column in the current-portfolio table.")

    ticker_col = "Ticker"
    current_col = "Current Allocations"
    asset_col = "Asset Group"

    # Try to find a band column (e.g. 'Min - Max %')
    band_col = None
    for c in df.columns:
        lc = c.lower()
        if "min" in lc and "max" in lc:
            band_col = c
            break

    # Keep just the relevant columns
    keep_cols = [ticker_col, asset_col, current_col]
    if band_col is not None:
        keep_cols.append(band_col)

    sub = df[keep_cols].copy()

    # Drop rows with no ticker or no current allocation
    sub = sub.dropna(subset=[ticker_col, current_col], how="any")

    # Drop any header echo / 'Total' rows
    sub = sub[
        (sub[ticker_col].astype(str).str.upper() != "TICKER")
        & (sub[ticker_col].astype(str).str.upper() != "TOTAL")
    ].copy()

    # Normalize tickers
    sub[ticker_col] = (
        sub[ticker_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Parse current weights (e.g. '29%' → 0.29)
    sub["Weight"] = sub[current_col].apply(_percent_to_float)

    # Parse min / max bands
    if band_col is not None:
        mins: list[float] = []
        maxs: list[float] = []
        for val in sub[band_col]:
            lo, hi = _band_to_min_max(val)
            mins.append(lo)
            maxs.append(hi)
        sub["MinPct"] = pd.to_numeric(mins, errors="coerce")
        sub["MaxPct"] = pd.to_numeric(maxs, errors="coerce")
    else:
        # No explicit bands → unconstrained per sleeve
        sub["MinPct"] = 0.0
        sub["MaxPct"] = 1.0

    # Ensure AssetGroup is a clean string
    sub["AssetGroup"] = sub[asset_col].astype(str).str.strip()

    # Build final output frame
    out = sub.rename(
        columns={
            ticker_col: "Ticker",
        }
    )[["Ticker", "AssetGroup", "MinPct", "MaxPct", "Weight"]]

    # Final clean-up: drop any blank-ticker rows
    out = out[out["Ticker"].astype(str).str.strip() != ""].reset_index(drop=True)

    return out


def load_investable_universe_from_case(case_path: Path) -> pd.DataFrame:
    """
    Load the full Investable Universe from the case Excel file.

    Returns a DataFrame with columns:
        Ticker, AssetGroup
    """

    # Read raw dataframe with no header
    df_raw = pd.read_excel(case_path, sheet_name=0, header=None)

    # Find the header row
    header_row = None
    for i, row in df_raw.iterrows():
        vals = row.astype(str).str.strip().str.lower()
        if "ticker" in vals.values or "symbol" in vals.values:
            header_row = i
            break

    if header_row is None:
        raise ValueError(
            f"Could not find a 'Ticker' or 'Symbol' header row in {case_path.name}"
        )

    # Re-read with proper header
    df = pd.read_excel(case_path, sheet_name=0, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    if "Ticker" not in df.columns:
        raise ValueError("Expected a 'Ticker' column in the investable universe sheet.")

    # Handle both possible names
    if "Asset Group" in df.columns:
        asset_col_name = "Asset Group"
    elif "Morningstar Category Group" in df.columns:
        asset_col_name = "Morningstar Category Group"
    else:
        raise ValueError(
            "Expected either 'Asset Group' or 'Morningstar Category Group' in the investable universe sheet."
        )

    # Keep only Ticker + Asset Group
    sub = df[["Ticker", asset_col_name]].copy()

    sub = sub.dropna(subset=["Ticker"])

    # Clean
    sub["Ticker"] = sub["Ticker"].astype(str).str.strip().str.upper()
    # Clean asset group + rename categories if desired
    sub["AssetGroup"] = (
        sub[asset_col_name]
        .astype(str)
        .str.strip()
        .replace({
            "Allocation": "Balanced / Allocation",
        })
    )

    # Drop duplicates on ticker
    sub = sub[["Ticker", "AssetGroup"]].drop_duplicates(subset=["Ticker"]).reset_index(drop=True)

    # Drop any blank tickers
    sub = sub[sub["Ticker"].astype(str).str.strip() != ""].reset_index(drop=True)

    return sub


if __name__ == "__main__":
    # Quick sanity check if you run: python case_io.py
    base_dir = Path(__file__).resolve().parent
    case_file = base_dir / "case_inputs.xlsx"

    cp_df = load_current_portfolio_from_case(case_file)
    print(cp_df)
    print("\nSum of weights:", cp_df["Weight"].sum())
