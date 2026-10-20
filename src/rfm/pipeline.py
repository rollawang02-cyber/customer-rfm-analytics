"""End-to-end analysis: ``python -m rfm.pipeline`` rebuilds every table and figure."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_mutual_info_score

from .clv import BetaGeoModel, GammaGammaModel, _purchase_days, customer_summary, holdout_purchases
from .config import CAL_END, CLV_HORIZON_DAYS, HOLDOUT_DAYS, PROCESSED_DIR, SNAPSHOT
from .data import clean_transactions, cleaning_audit, load_raw
from .features import build_rfm, replicate_tutorial_rfm
from .retention import churn_threshold, cohort_counts, cohort_retention, interpurchase_gaps, monthly_activity
from .segmentation import RFM_COLS, Segmenter, clusterable, evaluate_k, make_scaler, rfm_scores, surrogate_tree
from .sql import query_rfm


@dataclass
class Results:
    raw: pd.DataFrame
    tx: pd.DataFrame
    audit: pd.DataFrame
    tutorial: pd.DataFrame
    customers: pd.DataFrame
    k_selection: pd.DataFrame
    segmenter: Segmenter
    tree: object
    tree_fidelity: float
    gaps: pd.DataFrame
    retention: pd.DataFrame
    cohort_sizes: pd.DataFrame
    monthly: pd.DataFrame
    holdout: pd.DataFrame
    holdout_tracking: pd.DataFrame
    bgnbd: BetaGeoModel
    gamma_gamma: GammaGammaModel
    metrics: dict = field(default_factory=dict)


def _validate_clv(tx: pd.DataFrame):
    """Fit on data up to CAL_END, score the following 13 weeks."""
    obs_end = CAL_END.normalize() + pd.Timedelta(days=1)
    cal = customer_summary(tx, obs_end)
    actual = holdout_purchases(tx, obs_end, SNAPSHOT).reindex(cal.index, fill_value=0)
    bg = BetaGeoModel().fit(cal["x"], cal["t_x"], cal["T"])
    cal["actual"] = actual
    cal["bgnbd"] = bg.expected_purchases(HOLDOUT_DAYS, cal["x"], cal["t_x"], cal["T"])
    # Two transparent baselines. "repeat_rate" (repeat purchase days per day of tenure) is the fair one;
    # "run_rate" also counts the first purchase and therefore over-forecasts, especially for new customers.
    cal["repeat_rate"] = cal["x"] / cal["T"].clip(lower=1) * HOLDOUT_DAYS
    cal["run_rate"] = (cal["x"] + 1) / cal["T"].clip(lower=1) * HOLDOUT_DAYS

    days = np.arange(1, HOLDOUT_DAYS + 1)
    expected = [bg.expected_purchases(t, cal["x"], cal["t_x"], cal["T"]).sum() for t in days]
    daily = _purchase_days(tx, SNAPSHOT)
    daily = daily[(daily["day"] >= obs_end) & daily["CustomerID"].isin(cal.index)]
    actual_daily = daily.groupby("day").size().reindex(obs_end + pd.to_timedelta(days - 1, "D"), fill_value=0)
    tracking = pd.DataFrame({"day": actual_daily.index, "actual_cum": actual_daily.cumsum().to_numpy(),
                             "bgnbd_cum": expected, "repeat_rate_cum": cal["repeat_rate"].sum() * days / HOLDOUT_DAYS})
    scores = {"holdout_actual_total": int(cal["actual"].sum())}
    for name in ("bgnbd", "repeat_rate", "run_rate"):
        scores[f"holdout_{name}_total"] = float(cal[name].sum())
        scores[f"holdout_{name}_mae"] = float((cal[name] - cal["actual"]).abs().mean())
        scores[f"holdout_{name}_spearman"] = float(spearmanr(cal[name], cal["actual"])[0])
    scores.update({
        "calibration_params": {k: float(v) for k, v in bg.params_.items()},
    })
    return cal, tracking, scores


def gini(values) -> float:
    """Gini coefficient of a non-negative distribution (0 = equal, 1 = one holder)."""
    v = np.sort(np.clip(np.asarray(values, dtype=float), 0, None))
    n = len(v)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * v.sum()))


def run_analysis(raw: pd.DataFrame | None = None, n_boot: int = 20) -> Results:
    raw = load_raw() if raw is None else raw
    tx = clean_transactions(raw)
    audit = cleaning_audit(raw)
    tutorial = replicate_tutorial_rfm(raw)
    rfm = build_rfm(tx)

    # SQL layer must agree with pandas on the real data, not just the fixture.
    sql = query_rfm(raw)
    sql_gap = float((sql[RFM_COLS].sort_index() - rfm[RFM_COLS].sort_index()).abs().max().max())

    # --- segmentation ------------------------------------------------------
    X = make_scaler().fit_transform(clusterable(rfm)[RFM_COLS])
    k_sel = evaluate_k(X, n_boot=n_boot)
    seg = Segmenter(k=5).fit(rfm)
    customers = rfm.join(rfm_scores(rfm)).join(seg.predict(rfm))
    customers["segment"] = customers["segment"].cat.add_categories("Net spend <= 0")
    customers["segment"] = customers["segment"].fillna("Net spend <= 0")
    labelled = customers[customers["segment"] != "Net spend <= 0"]
    tree, fidelity = surrogate_tree(rfm, labelled["segment"].cat.remove_unused_categories())

    # --- repeat purchase cycle & status -------------------------------------
    gaps = interpurchase_gaps(tx).merge(customers[["segment"]], left_on="CustomerID", right_index=True)
    active_cut = float(gaps["gap_days"].median())
    lapsed_cut = churn_threshold(gaps, 0.9)
    customers["status"] = pd.cut(
        customers["recency_days"], [-np.inf, active_cut, lapsed_cut, np.inf],
        labels=[f"Active (<= {active_cut:.0f}d)", f"Cooling ({active_cut:.0f}-{lapsed_cut:.0f}d)",
                f"Lapsed (> {lapsed_cut:.0f}d)"],
    )

    # --- cohorts -------------------------------------------------------------
    counts = cohort_counts(tx)
    retention = cohort_retention(tx)
    monthly = monthly_activity(tx)
    customers["cohort_month"] = customers["first_purchase"].dt.to_period("M")

    # --- CLV -----------------------------------------------------------------
    holdout, tracking, clv_scores = _validate_clv(tx)
    summ = customer_summary(tx, SNAPSHOT)
    bg = BetaGeoModel().fit(summ["x"], summ["t_x"], summ["T"])
    repeat = summ[summ["x"] > 0]
    gg = GammaGammaModel().fit(repeat["x"], repeat["monetary_value"])
    summ["p_alive"] = bg.p_alive(summ["x"], summ["t_x"], summ["T"])
    summ["exp_purchases"] = bg.expected_purchases(CLV_HORIZON_DAYS, summ["x"], summ["t_x"], summ["T"])
    summ["exp_spend"] = gg.expected_spend(summ["x"], summ["monetary_value"])
    summ["clv"] = summ["exp_purchases"] * summ["exp_spend"]
    customers = customers.join(summ)
    # Customers whose every purchase day was cancelled have no transactions left.
    customers[["exp_purchases", "clv"]] = customers[["exp_purchases", "clv"]].fillna(0.0)
    customers["clv_tier"] = pd.qcut(customers["clv"].rank(method="first", ascending=False),
                                    [0, 0.2, 0.6, 1.0], labels=["Top 20% CLV", "Middle 40%", "Bottom 40%"])

    seg_rev = labelled.groupby("segment", observed=True)["monetary"].sum()
    gross, net = rfm["gross_revenue"].sum(), rfm["monetary"].sum()
    m_sorted = rfm["monetary"].clip(lower=0).sort_values(ascending=False)
    metrics = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(tx)),
        "customers": int(len(rfm)),
        "customers_clustered": int(len(labelled)),
        "net_revenue": float(net),
        "gross_revenue": float(gross),
        "return_rate_value": float(1 - net / gross),
        "guest_revenue_share": float(-audit.loc[2, "revenue_change"] / audit.loc[1, "revenue"]),
        "one_time_buyer_share": float((rfm["frequency"] == 1).mean()),
        "gini": gini(rfm["monetary"]),
        "top20_revenue_share": float(m_sorted.iloc[: int(len(m_sorted) * 0.2)].sum() / m_sorted.sum()),
        "top1_revenue_share": float(m_sorted.iloc[: int(len(m_sorted) * 0.01)].sum() / m_sorted.sum()),
        "tutorial_mean_monetary": float(tutorial["Sales"].mean()),
        "tutorial_revenue_captured": float(tutorial["Sales"].sum() / rfm["gross_revenue"].sum()),
        "corrected_mean_monetary": float(rfm["monetary"].mean()),
        "sql_pandas_max_abs_diff": sql_gap,
        "median_gap_days": active_cut,
        "p90_gap_days": lapsed_cut,
        "silhouette_k5": float(k_sel.loc[5, "silhouette"]),
        "stability_ari_k5": float(k_sel.loc[5, "stability_ari"]),
        "tree_fidelity": fidelity,
        "kmeans_vs_rfm_ami": float(adjusted_mutual_info_score(labelled["segment"], labelled["rfm_segment"])),
        "segment_sizes": labelled["segment"].value_counts().reindex(seg_rev.index).astype(int).to_dict(),
        "segment_revenue_share": (seg_rev / seg_rev.sum()).round(4).to_dict(),
        "bgnbd_params": {k: float(v) for k, v in bg.params_.items()},
        "gamma_gamma_params": {k: float(v) for k, v in gg.params_.items()},
        "gg_freq_spend_corr": float(np.corrcoef(repeat["x"], repeat["monetary_value"])[0, 1]),
        "share_p_alive_above_0_95": float((summ["p_alive"] > 0.95).mean()),
        "clv_total_180d": float(summ["clv"].sum()),
        **clv_scores,
    }
    return Results(raw, tx, audit, tutorial, customers, k_sel, seg, tree, fidelity, gaps, retention,
                   counts[[0]].rename(columns={0: "cohort_size"}), monthly, holdout, tracking, bg, gg, metrics)


def write_outputs(res: Results) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cust = res.customers.copy()
    cust["cohort_month"] = cust["cohort_month"].astype(str)
    num = cust.select_dtypes("number").columns
    cust[num] = cust[num].round(4)
    cust.to_csv(PROCESSED_DIR / "customers.csv")
    ret = res.retention.copy()
    ret.index = ret.index.astype(str)
    ret.round(4).to_csv(PROCESSED_DIR / "cohort_retention.csv")
    mon = res.monthly.copy()
    mon.index = mon.index.astype(str)
    mon.round(2).to_csv(PROCESSED_DIR / "monthly_activity.csv")
    res.k_selection.round(4).to_csv(PROCESSED_DIR / "k_selection.csv")
    res.audit.round(2).to_csv(PROCESSED_DIR / "cleaning_audit.csv", index=False)
    res.holdout_tracking.round({"bgnbd_cum": 2}).to_csv(PROCESSED_DIR / "clv_holdout_tracking.csv", index=False)
    (PROCESSED_DIR / "metrics.json").write_text(json.dumps(res.metrics, indent=2, default=float))


def main() -> None:
    from .viz.charts import make_all_figures

    res = run_analysis()
    write_outputs(res)
    paths = make_all_figures(res)
    print(json.dumps(res.metrics, indent=2, default=float))
    print(f"\n{len(paths)} figures written.")


if __name__ == "__main__":
    main()
