"""Loading and cleaning the UCI Online Retail transactions.

Cleaning is expressed as an ordered list of named steps so that the same rules
drive both the clean table and the audit trail shown in the report.
"""
from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from .config import NON_MERCH_CODES, NON_MERCH_PREFIXES, RAW_PATH

RAW_DTYPES = {
    "InvoiceNo": str,
    "StockCode": str,
    "Description": str,
    "CustomerID": str,
    "Country": str,
}


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV (plain or zipped) with IDs kept as strings."""
    df = pd.read_csv(path, dtype=RAW_DTYPES, encoding="utf-8")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%m/%d/%Y %H:%M")
    return df


def _is_non_merch(codes: pd.Series) -> pd.Series:
    return codes.isin(NON_MERCH_CODES) | codes.str.startswith(NON_MERCH_PREFIXES)


Step = tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]

CLEANING_STEPS: list[Step] = [
    ("Exact duplicate lines", lambda d: d.drop_duplicates()),
    ("No CustomerID (guest checkout)", lambda d: d[d["CustomerID"].notna()]),
    ("UnitPrice <= 0 (free / write-off)", lambda d: d[d["UnitPrice"] > 0]),
    ("Non-merchandise codes (postage, fees, adjustments)", lambda d: d[~_is_non_merch(d["StockCode"])]),
]


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply every cleaning step and add `Revenue` and `IsCancel`.

    Returns (``C``-prefixed invoices) are kept with negative revenue so that
    Monetary can be computed net of returns instead of silently ignoring them.
    """
    df = raw
    for _, step in CLEANING_STEPS:
        df = step(df)
    df = df.copy()
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]
    df["IsCancel"] = df["InvoiceNo"].str.startswith("C")
    return df.reset_index(drop=True)


def cleaning_audit(raw: pd.DataFrame) -> pd.DataFrame:
    """Rows and net revenue remaining after each cleaning step."""
    records = [{"step": "Raw extract", "rows": len(raw), "revenue": float((raw["Quantity"] * raw["UnitPrice"]).sum())}]
    df = raw
    for name, step in CLEANING_STEPS:
        df = step(df)
        records.append({"step": name, "rows": len(df), "revenue": float((df["Quantity"] * df["UnitPrice"]).sum())})
    out = pd.DataFrame(records)
    out["rows_removed"] = (-out["rows"].diff()).fillna(0).astype(int)
    out["revenue_change"] = out["revenue"].diff().fillna(0.0)
    return out
