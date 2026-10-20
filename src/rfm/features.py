"""Customer-level RFM feature table."""
from __future__ import annotations

import pandas as pd

from .config import SNAPSHOT


def build_rfm(tx: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    """One row per customer with RFM plus the context needed to interpret it.

    * ``recency_days``  - days from the last purchase date to ``snapshot``
    * ``frequency``     - distinct purchase invoices (returns excluded)
    * ``monetary``      - purchases minus returns (net revenue, GBP)
    * ``gross_revenue`` / ``returned_value`` / ``return_rate``
    * ``aov``           - gross revenue per purchase invoice
    * ``tenure_days``   - days since the first purchase
    * ``country``       - the customer's most frequent invoice country

    Customers whose only activity is returns (purchases pre-date the extract)
    have no purchase row and are therefore not in the table.
    """
    buys = tx[~tx["IsCancel"]]
    g = buys.groupby("CustomerID")
    out = pd.DataFrame(
        {
            "first_purchase": g["InvoiceDate"].min(),
            "last_purchase": g["InvoiceDate"].max(),
            "frequency": g["InvoiceNo"].nunique(),
            "gross_revenue": g["Revenue"].sum(),
            "country": g["Country"].agg(lambda s: s.mode().iloc[0]),
        }
    )
    returned = -tx[tx["IsCancel"]].groupby("CustomerID")["Revenue"].sum()
    out["returned_value"] = returned.reindex(out.index, fill_value=0.0)
    out["monetary"] = out["gross_revenue"] - out["returned_value"]
    out["recency_days"] = (snapshot - out["last_purchase"].dt.normalize()).dt.days
    out["tenure_days"] = (snapshot - out["first_purchase"].dt.normalize()).dt.days
    out["aov"] = out["gross_revenue"] / out["frequency"]
    out["return_rate"] = out["returned_value"] / out["gross_revenue"]
    out.index.name = "CustomerID"
    cols = [
        "recency_days", "frequency", "monetary", "gross_revenue", "returned_value",
        "return_rate", "aov", "tenure_days", "first_purchase", "last_purchase", "country",
    ]
    return out[cols]


def replicate_tutorial_rfm(raw: pd.DataFrame) -> pd.DataFrame:
    """Faithful re-implementation of the reference tutorial, bugs included.

    The tutorial does ``df_f = df`` followed by an *in-place*
    ``drop_duplicates(subset=['InvoiceNo', 'CustomerID'])``. Because ``df_f``
    is an alias, the de-duplication also hits ``df``: only the first line of
    every invoice survives, and Monetary is summed over those first lines only.
    Recency is measured from 2012-01-01. Returns are dropped, never netted.
    """
    df = raw.drop_duplicates()
    df = df[(df["UnitPrice"] >= 0) & (df["Quantity"] > 0)]
    df = df[df["CustomerID"].notna()].copy()
    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    df = df.drop_duplicates(subset=["InvoiceNo", "CustomerID"], keep="first")  # the aliasing bug
    g = df.groupby("CustomerID")
    return pd.DataFrame(
        {
            "Sales": g["Sales"].sum(),
            "DateDiff": (pd.Timestamp("2012-01-01") - g["InvoiceDate"].max()).dt.days,
            "Frequency": g["InvoiceNo"].nunique(),
        }
    )
