"""Cohorts, repeat-purchase cycles and monthly activity."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _purchase_months(tx: pd.DataFrame) -> pd.DataFrame:
    buys = tx.loc[~tx["IsCancel"], ["CustomerID", "InvoiceDate"]]
    pm = buys.assign(order_month=buys["InvoiceDate"].dt.to_period("M"))[["CustomerID", "order_month"]]
    pm = pm.drop_duplicates()
    pm["cohort"] = pm.groupby("CustomerID")["order_month"].transform("min")
    pm["period"] = (pm["order_month"] - pm["cohort"]).apply(lambda d: d.n)
    return pm


def cohort_counts(tx: pd.DataFrame) -> pd.DataFrame:
    """Active customers per (acquisition month x months since acquisition)."""
    pm = _purchase_months(tx)
    counts = pm.pivot_table(index="cohort", columns="period", values="CustomerID", aggfunc="nunique")
    last = pm["order_month"].max()
    counts = counts.reindex(columns=range((last - counts.index.min()).n + 1)).astype(float)
    # Observable cells (cohort + period <= last month) with no buyers are 0;
    # cells beyond the end of the data stay NaN (right-censored).
    horizon = np.array([(last - c).n for c in counts.index])[:, None]
    observable = counts.columns.to_numpy()[None, :] <= horizon
    return counts.where(~observable, counts.fillna(0.0))


def cohort_retention(tx: pd.DataFrame) -> pd.DataFrame:
    """Share of each acquisition cohort that purchases in month k (k=0 is 100%)."""
    counts = cohort_counts(tx)
    return counts.div(counts[0], axis=0)


def interpurchase_gaps(tx: pd.DataFrame) -> pd.DataFrame:
    """Days between consecutive distinct purchase days, one row per gap."""
    days = (
        tx.loc[~tx["IsCancel"], ["CustomerID", "InvoiceDate"]]
        .assign(day=lambda d: d["InvoiceDate"].dt.normalize())[["CustomerID", "day"]]
        .drop_duplicates()
        .sort_values(["CustomerID", "day"])
    )
    days["gap_days"] = days.groupby("CustomerID")["day"].diff().dt.days
    return days.dropna(subset=["gap_days"])[["CustomerID", "day", "gap_days"]].reset_index(drop=True)


def monthly_activity(tx: pd.DataFrame) -> pd.DataFrame:
    """Active customers split into new vs returning, plus net revenue, by month."""
    pm = _purchase_months(tx)
    active = pm.groupby("order_month").agg(
        active=("CustomerID", "nunique"),
        new=("period", lambda p: int((p == 0).sum())),
    )
    active["returning"] = active["active"] - active["new"]
    revenue = tx.assign(order_month=tx["InvoiceDate"].dt.to_period("M")).groupby("order_month")["Revenue"].sum()
    active["net_revenue"] = revenue
    return active


def churn_threshold(gaps: pd.DataFrame, quantile: float = 0.9) -> float:
    """Inactivity (days) beyond which a repeat customer is unusually overdue.

    Using the 90th percentile of observed repeat gaps makes the churn rule
    data-driven rather than an arbitrary "90 days".
    """
    return float(np.quantile(gaps["gap_days"], quantile))
