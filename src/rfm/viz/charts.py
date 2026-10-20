"""All report figures. Each function takes analysis outputs and returns a Figure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy.stats import gaussian_kde

from ..segmentation import SEGMENT_ORDER
from .theme import (AXIS, BLUE_RAMP, CATEGORICAL, DIVERGING, GRID, INK, INK_2, MUTED, NEUTRAL,
                    SEGMENT_COLORS, SEQUENTIAL, SURFACE, apply_theme, headline, save, source_note)

apply_theme()

GBP = FuncFormatter(lambda v, _: f"£{v / 1e6:.1f}M" if abs(v) >= 1e6 else (f"£{v / 1e3:.0f}k" if abs(v) >= 1e3 else f"£{v:.0f}"))
LOG_TICKS = {1: "1", 2: "2", 5: "5", 10: "10", 20: "20", 50: "50", 100: "100", 200: "200", 500: "500",
             1000: "1k", 2000: "2k", 5000: "5k", 10000: "10k", 50000: "50k", 100000: "100k"}


def _labelled(customers: pd.DataFrame) -> pd.DataFrame:
    c = customers[customers["segment"].isin(SEGMENT_ORDER)].copy()
    c["segment"] = pd.Categorical(c["segment"].astype(str), SEGMENT_ORDER, ordered=True)
    return c


def _log_axis(ax, axis="y", lo=1, hi=1e5, money=False, decades=False):
    ticks = [t for t in LOG_TICKS if lo <= t <= hi and (not decades or np.log10(t).is_integer())]
    labels = [("£" if money else "") + LOG_TICKS[t] for t in ticks]
    if axis == "y":
        ax.set_yticks(np.log10(ticks), labels)
    else:
        ax.set_xticks(np.log10(ticks), labels)


# 01 -------------------------------------------------------------------------
def fig_data_audit(audit: pd.DataFrame, tutorial: pd.DataFrame, customers: pd.DataFrame, metrics: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.55})
    steps = audit.iloc[1:].copy()
    names = ["Exact duplicates", "No CustomerID", "UnitPrice ≤ 0", "Postage, fees,\nadjustments"]
    y = np.arange(len(steps))[::-1]

    ax = axes[0]
    ax.barh(y, steps["rows_removed"], color=CATEGORICAL[0], height=0.56)
    for yi, v in zip(y, steps["rows_removed"]):
        ax.text(v + 2500, yi, f"{v:,}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(0, steps["rows_removed"].max() * 1.3)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e3:.0f}k"))
    ax.grid(axis="y", visible=False)
    ax.set_title("a  Lines removed per cleaning rule")
    ax.set_xlabel(f"{metrics['raw_rows']:,} raw lines → {metrics['clean_rows']:,} analysed")

    ax = axes[1]
    ax.barh(y, -steps["revenue_change"], color=CATEGORICAL[0], height=0.56)
    for yi, v in zip(y, -steps["revenue_change"]):
        ax.text(max(v, 0) + 25000, yi, GBP(abs(v), None) if abs(v) >= 1 else "£0", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlim(0, (-steps["revenue_change"]).max() * 1.3)
    ax.xaxis.set_major_formatter(GBP)
    ax.grid(axis="y", visible=False)
    ax.set_title("b  Net revenue removed per rule")
    ax.set_xlabel(f"Guest checkouts alone hold {metrics['guest_revenue_share']:.1%} of revenue")

    ax = axes[2]
    j = tutorial.join(customers[["gross_revenue"]], how="inner")
    j = j[(j["Sales"] > 0) & (j["gross_revenue"] > 0)]
    ax.scatter(np.log10(j["gross_revenue"]), np.log10(j["Sales"]), s=9, color=CATEGORICAL[0], alpha=0.35,
               linewidths=0)
    ax.plot([0, 5.6], [0, 5.6], color=MUTED, lw=1)
    ax.text(4.55, 4.85, "y = x", color=MUTED, fontsize=8.5, rotation=38)
    _log_axis(ax, "x", 1, 3e5, money=True, decades=True)
    _log_axis(ax, "y", 1, 3e5, money=True, decades=True)
    ax.set_xlim(0.4, 5.6)
    ax.set_ylim(0, 5.6)
    ax.set_xlabel("Correct gross spend per customer")
    ax.set_ylabel("Tutorial 'Sales' per customer")
    ax.set_title("c  The tutorial keeps one line per invoice")
    ax.text(0.5, 5.25, f"Tutorial Monetary captures {metrics['tutorial_revenue_captured']:.1%} of revenue\n"
                       f"mean £{metrics['tutorial_mean_monetary']:,.0f} vs £{metrics['corrected_mean_monetary']:,.0f} (net)",
            fontsize=8.5, color=INK, va="top")
    headline(fig, "Data audit: one in four lines has no customer, and the reference tutorial's Monetary is wrong",
             "Every cleaning rule is a named, testable step; returns are kept and netted rather than dropped.")
    source_note(fig)
    return fig


# 02 -------------------------------------------------------------------------
def fig_revenue_concentration(customers: pd.DataFrame, metrics: dict):
    m = customers["monetary"].clip(lower=0).sort_values(ascending=False).to_numpy()
    x = np.arange(1, len(m) + 1) / len(m)
    y = np.cumsum(m) / m.sum()
    gini = 1 - 2 * np.trapezoid(np.concatenate([[0], np.cumsum(m[::-1]) / m.sum()]), np.linspace(0, 1, len(m) + 1))
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ax.fill_between(x, x, y, color=BLUE_RAMP[1], alpha=0.8, linewidth=0)
    ax.plot(x, y, color=CATEGORICAL[0])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1)
    for share in (0.01, 0.2, 0.5):
        yi = y[int(share * len(m)) - 1]
        ax.plot(share, yi, "o", ms=7, color=CATEGORICAL[0], mec=SURFACE, mew=2)
        ax.text(share + 0.02, yi - 0.035, f"Top {share:.0%} of customers → {yi:.0%} of revenue", fontsize=8.5,
                color=INK, va="top")
    ax.text(0.62, 0.42, "Perfect equality", color=MUTED, fontsize=8.5, rotation=35)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Customers, ranked by net spend")
    ax.set_ylabel("Cumulative share of net revenue")
    headline(fig, f"Revenue is extremely concentrated (Gini = {gini:.2f})",
             "Lorenz curve of net revenue across 4,334 identified customers.")
    source_note(fig)
    return fig, float(gini)


# 03 -------------------------------------------------------------------------
def fig_monthly_activity(monthly: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.6), sharex=True, gridspec_kw={"hspace": 0.32, "height_ratios": [1.3, 1]})
    labels = [p.strftime("%b\n%Y") if p.month in (12, 1) else p.strftime("%b") for p in monthly.index]
    x = np.arange(len(monthly))
    partial = monthly.index.astype(str) == "2011-12"
    hatch = ["////" if p else "" for p in partial]
    kw = dict(width=0.7, hatch=hatch, linewidth=0)
    a1.bar(x, monthly["returning"], color=CATEGORICAL[0], label="Returning", **kw)
    a1.bar(x, monthly["new"], bottom=monthly["returning"] + 8, color=CATEGORICAL[2], label="New", **kw)
    a1.legend(loc="upper left", ncols=2)
    a1.set_ylabel("Active customers")
    a1.set_title("a  Active customers per month - growth after Aug 2011 is driven by returning buyers")
    a1.annotate("All 'new' by construction\n(the extract starts here)", (x[0], monthly["active"].iloc[0]),
                xytext=(x[0] + 0.9, monthly["active"].iloc[0] + 260), fontsize=7.8, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    a2.bar(x, monthly["net_revenue"], color=INK_2, **kw)
    for xi in x[partial]:
        a1.text(xi, monthly["active"].iloc[-1] + 40, "1-9 Dec\nonly", ha="center", fontsize=7.8, color=INK_2)
        a2.text(xi, monthly["net_revenue"].iloc[-1] + 30000, "1-9 Dec\nonly", ha="center", fontsize=7.8, color=INK_2)
    a2.yaxis.set_major_formatter(GBP)
    a2.set_ylabel("Net revenue")
    a2.set_title("b  Net revenue per month - Q4 seasonality (gift wholesale)")
    a2.set_xticks(x, labels)
    for ax in (a1, a2):
        ax.grid(axis="x", visible=False)
    rev = monthly["net_revenue"]
    lift = rev.loc[[pd.Period(f"2011-{m:02d}", "M") for m in (9, 10, 11)]].sum() / \
        rev.loc[[pd.Period(f"2011-{m:02d}", "M") for m in (6, 7, 8)]].sum() - 1
    headline(fig, f"Sep-Nov revenue runs {lift:.0%} above Jun-Aug; December 2011 is a partial month, not a drop")
    source_note(fig)
    return fig


# 04 -------------------------------------------------------------------------
def fig_cohort_heatmap(retention: pd.DataFrame, sizes: pd.DataFrame):
    ret = retention.iloc[:-1, 1:13]  # drop the 9-day Dec-2011 cohort and month 0 (always 100%)
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    data = ret.to_numpy(dtype=float)
    im = ax.imshow(np.ma.masked_invalid(data), cmap=SEQUENTIAL, vmin=0, vmax=0.45, aspect="auto")
    last = retention.index.max()
    for i, cohort in enumerate(ret.index):
        for j, period in enumerate(ret.columns):
            v = data[i, j]
            if np.isnan(v):
                continue
            is_partial = (cohort + period) == last
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color=INK if is_partial else (SURFACE if v > 0.28 else INK), alpha=0.7 if is_partial else 1)
            if is_partial:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", edgecolor=SURFACE, lw=0))
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SURFACE, lw=2))
    ylabels = [f"{c.strftime('%b %Y')}{'  (legacy*)' if i == 0 else ''}   n={int(sizes.loc[c, 'cohort_size']):,}"
               for i, c in enumerate(ret.index)]
    ax.set_yticks(range(len(ret)), ylabels)
    ax.set_xticks(range(ret.shape[1]), [f"+{p}" for p in ret.columns])
    ax.set_xlabel("Months since first purchase")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, format=PercentFormatter(1, decimals=0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2)
    cb.set_label("Share of cohort purchasing", color=INK_2, fontsize=8.5)
    headline(fig, "The Dec-2010 'cohort' retains far better than any other - because it is not a real cohort",
             "Monthly acquisition cohorts. Dec 2010 holds every pre-existing customer (left-censored); hatched cells are the 9-day Dec 2011.")
    source_note(fig, "* Left-censored: the extract starts 2010-12-01, so long-standing wholesale accounts are labelled 'new' in Dec 2010. "
                     "Source: UCI Online Retail.")
    return fig


# 05 -------------------------------------------------------------------------
def fig_k_selection(k_sel: pd.DataFrame, chosen: int = 5):
    panels = [("silhouette", "Silhouette ↑", None), ("davies_bouldin", "Davies-Bouldin ↓", None),
              ("calinski_harabasz", "Calinski-Harabasz ↑", None), ("stability_ari", "Bootstrap stability (ARI) ↑", "stability_ari_p05")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), gridspec_kw={"wspace": 0.35})
    for ax, (col, title, low) in zip(axes, panels):
        ax.axvspan(chosen - 0.4, chosen + 0.4, color=NEUTRAL, zorder=0)
        if low:
            ax.vlines(k_sel.index, k_sel[low], k_sel[col], color=BLUE_RAMP[3], lw=2)
        ax.plot(k_sel.index, k_sel[col], color=CATEGORICAL[0], marker="o", ms=7, mec=SURFACE, mew=2)
        ax.set_title(title)
        ax.set_xticks(k_sel.index)
        ax.set_xlabel("k")
        ax.grid(axis="x", visible=False)
        v = k_sel.loc[chosen, col]
        ax.annotate(f"{v:,.3f}" if v < 100 else f"{v:,.0f}", (chosen, v), xytext=(8, 8), textcoords="offset points",
                    fontsize=8.5, color=INK)
    axes[3].text(2.1, k_sel["stability_ari_p05"].min(), "bar: mean ARI down to\nits 5th percentile", fontsize=7.8, color=INK_2)
    headline(fig, "No k yields strong structure; k = 5 is chosen for actionability, with the best Davies-Bouldin among k ≥ 4",
             "K-means on log1p + z-scored R, F, M (n = 4,322). Silhouette < 0.35 for k ≥ 3: segments partition a continuum. "
             "k = 5 mean stability ≈ k = 4, but with a wider bootstrap spread.")
    source_note(fig)
    return fig


# 06 -------------------------------------------------------------------------
def fig_segment_violins(customers: pd.DataFrame):
    c = _labelled(customers)
    specs = [("recency_days", "Recency (days since last order)", 1, 400, False),
             ("frequency", "Frequency (orders)", 1, 200, False),
             ("monetary", "Monetary (net spend)", 1, 3e5, True)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"wspace": 0.28})
    for ax, (col, title, lo, hi, money) in zip(axes, specs):
        for i, seg in enumerate(SEGMENT_ORDER):
            v = np.log10(c.loc[c["segment"] == seg, col].clip(lower=1))
            parts = ax.violinplot(v, positions=[i], widths=0.82, showextrema=False, bw_method=0.25)
            for body in parts["bodies"]:
                body.set_facecolor(SEGMENT_COLORS[seg])
                body.set_edgecolor(SEGMENT_COLORS[seg])
                body.set_alpha(0.55)
                body.set_linewidth(1)
            q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
            ax.vlines(i, q1, q3, color=INK, lw=3)
            ax.plot(i, med, "o", ms=6.5, color=SURFACE, mec=INK, mew=1.4, zorder=3)
            ax.text(i + 0.12, med, f"{10 ** med:,.0f}", fontsize=7.8, color=INK, va="center")
        _log_axis(ax, "y", lo, hi, money)
        ax.set_xticks(range(5), [s.replace(" / ", "/\n") for s in SEGMENT_ORDER], fontsize=8.5)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    headline(fig, "Frequency and Recency drive the split; Loyal and At Risk differ mostly in how recently they bought",
             "Violin = density on log scale; bar = interquartile range; dot = median (labelled).")
    source_note(fig)
    return fig


# 07 -------------------------------------------------------------------------
def fig_segment_profile(customers: pd.DataFrame):
    c = _labelled(customers)
    g = c.groupby("segment", observed=True)
    prof = pd.DataFrame({
        "Recency\n(days)": g["recency_days"].median(),
        "Orders": g["frequency"].median(),
        "Net spend": g["monetary"].median(),
        "Avg order\nvalue": g["aov"].median(),
        "Tenure\n(days)": g["tenure_days"].median(),
        "Made a\nreturn": g["returned_value"].apply(lambda v: (v > 0).mean()),
        "Outside\nUK": g["country"].apply(lambda s: (s != "United Kingdom").mean()),
        "Active\n≤ 28d": g["status"].apply(lambda s: s.astype(str).str.startswith("Active").mean()),
        "180-day\nCLV": g["clv"].median(),
    })
    logged = prof.copy()
    for col in ["Recency\n(days)", "Orders", "Net spend", "Avg order\nvalue", "180-day\nCLV"]:
        logged[col] = np.log(prof[col])
    z = (logged - logged.mean()) / logged.std(ddof=0)
    fmt = {"Recency\n(days)": "{:.0f}", "Orders": "{:.0f}", "Net spend": "£{:,.0f}", "Avg order\nvalue": "£{:,.0f}",
           "Tenure\n(days)": "{:.0f}", "Made a\nreturn": "{:.0%}", "Outside\nUK": "{:.0%}", "Active\n≤ 28d": "{:.0%}",
           "180-day\nCLV": "£{:,.0f}"}
    fig, ax = plt.subplots(figsize=(13, 4.6))
    im = ax.imshow(z.to_numpy(), cmap=DIVERGING, vmin=-1.8, vmax=1.8, aspect="auto")
    for i in range(z.shape[0]):
        for j, col in enumerate(z.columns):
            zz = z.iat[i, j]
            ax.text(j, i, fmt[col].format(prof.iat[i, j]), ha="center", va="center", fontsize=9,
                    color=SURFACE if abs(zz) > 1.15 else INK, fontweight="medium")
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SURFACE, lw=2.5))
    ax.set_xticks(range(z.shape[1]), z.columns, fontsize=8.5)
    ax.set_yticks(range(z.shape[0]), z.index)
    ax.xaxis.tick_top()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for t, seg in zip(ax.get_yticklabels(), z.index):
        t.set_color(INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015)
    cb.outline.set_visible(False)
    cb.set_label("Column z-score (log for £, counts)", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7.5, colors=INK_2)
    headline(fig, "Segment profiles: Champions buy 3x more often than Loyal, At Risk were mid-value and went quiet",
             "Cell text = segment median (or share); colour = how far the segment sits from the cross-segment mean. Red = higher, blue = lower.")
    source_note(fig)
    return fig


# 08 -------------------------------------------------------------------------
def _rule(feature: str, thr: float) -> str:
    if feature == "recency_days":
        return f"Recency ≤ {int(np.floor(thr))} d"
    if feature == "frequency":
        return f"Orders ≤ {int(np.floor(thr))}"
    return f"Net spend ≤ £{thr:,.0f}"


def fig_segment_tree(tree, fidelity: float, features=("recency_days", "frequency", "monetary")):
    t = tree.tree_
    classes = list(tree.classes_)
    order = [classes.index(s) for s in SEGMENT_ORDER]
    pos, leaves = {}, []

    def walk(node, depth):
        if t.children_left[node] == -1:
            pos[node] = (len(leaves), depth)
            leaves.append(node)
            return pos[node][0]
        xl = walk(t.children_left[node], depth + 1)
        xr = walk(t.children_right[node], depth + 1)
        pos[node] = ((xl + xr) / 2, depth)
        return pos[node][0]

    walk(0, 0)
    max_depth = max(d for _, d in pos.values())
    fig, ax = plt.subplots(figsize=(16, 7.4))
    ax.set_xlim(-0.6, len(leaves) - 0.4)
    ax.set_ylim(max_depth + 0.75, -0.55)
    ax.axis("off")
    w, h = 0.86, 0.52

    for node, (x, d) in pos.items():
        for child, tag in ((t.children_left[node], "yes"), (t.children_right[node], "no")):
            if child == -1:
                continue
            cx, cd = pos[child]
            verts = [(x, d + h / 2), (x, d + 0.5 * (1 + h / 2) + 0.05), (cx, cd - 0.5 * (1 - h / 2) - 0.05), (cx, cd - h / 2)]
            ax.add_patch(PathPatch(MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]),
                                   fill=False, edgecolor=AXIS, lw=1.3))
            ax.text((x + cx) / 2, (d + cd) / 2 + 0.02, tag, fontsize=7.8, color=MUTED, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))

    for node, (x, d) in pos.items():
        counts = t.value[node][0]
        counts = counts / counts.sum()
        n = int(t.n_node_samples[node])
        leaf = t.children_left[node] == -1
        ax.add_patch(FancyBboxPatch((x - w / 2, d - h / 2), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc=SURFACE, ec=AXIS if not leaf else INK_2, lw=1 if not leaf else 1.2))
        if leaf:
            pred = classes[int(np.argmax(counts))]
            ax.add_patch(Rectangle((x - w / 2 + 0.06, d - h / 2 + 0.08), 0.05, 0.13, color=SEGMENT_COLORS[pred]))
            ax.text(x - w / 2 + 0.15, d - h / 2 + 0.145, pred, fontsize=8.6, fontweight="bold", color=INK, va="center")
            ax.text(x - w / 2 + 0.06, d - 0.01, f"n = {n:,} · purity {counts.max():.0%}", fontsize=7.6, color=INK_2, va="center")
        else:
            ax.text(x, d - h / 2 + 0.14, _rule(features[t.feature[node]], t.threshold[node]), fontsize=9,
                    fontweight="bold", color=INK, ha="center", va="center")
            ax.text(x, d - 0.01, f"n = {n:,}", fontsize=7.6, color=INK_2, ha="center", va="center")
        left = x - w / 2 + 0.06
        for k in order:
            share = counts[k]
            if share <= 0:
                continue
            ax.add_patch(Rectangle((left, d + h / 2 - 0.14), (w - 0.12) * share, 0.075,
                                   color=SEGMENT_COLORS[classes[k]], lw=0))
            left += (w - 0.12) * share
    handles = [Rectangle((0, 0), 1, 1, color=SEGMENT_COLORS[s]) for s in SEGMENT_ORDER]
    ax.legend(handles, SEGMENT_ORDER, loc="lower center", ncols=5, bbox_to_anchor=(0.5, -0.06), fontsize=8.5,
              handlelength=1, handleheight=1)
    headline(fig, f"Three questions recover the K-means segments for {fidelity:.0%} of customers",
             "Depth-3 surrogate decision tree on raw R, F, M (5-fold CV fidelity). Bars show the segment mix in each node.")
    source_note(fig)
    return fig


# 09 -------------------------------------------------------------------------
RFM_RULE_ORDER = ["Champions", "Loyal", "Potential Loyalist", "New", "Promising", "Need Attention",
                  "About to Sleep", "At Risk", "Can't Lose", "Hibernating"]


def fig_kmeans_vs_rules(customers: pd.DataFrame, ami: float):
    c = _labelled(customers)
    ct = pd.crosstab(c["segment"], c["rfm_segment"]).reindex(columns=RFM_RULE_ORDER, fill_value=0)
    share = ct.div(ct.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 4.6))
    im = ax.imshow(share.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=0.8, aspect="auto")
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.iat[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5, color=SURFACE if v > 0.45 else INK)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=SURFACE, lw=2))
    ax.set_xticks(range(share.shape[1]), [s.replace(" ", "\n") for s in share.columns], fontsize=8.5)
    ax.set_yticks(range(share.shape[0]), [f"{s}  (n={ct.loc[s].sum():,})" for s in share.index])
    ax.xaxis.tick_top()
    ax.set_xlabel("Rule-based RFM segment (R x F quintile map)")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015, format=PercentFormatter(1, decimals=0))
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=INK_2)
    headline(fig, f"Data-driven and rule-based segments agree only moderately (AMI = {ami:.2f})",
             "Row = K-means segment, cell = share of that segment falling in each rule-based segment. The rule map ignores Monetary entirely.")
    source_note(fig)
    return fig


# 10 -------------------------------------------------------------------------
def behaviour_flags(customers: pd.DataFrame, median_gap: float, p90_gap: float) -> pd.DataFrame:
    c = customers[customers["monetary"] > 0]
    return pd.DataFrame({
        "Top-20% spend": c["monetary"] >= c["monetary"].quantile(0.8),
        "Repeat buyer": c["frequency"] >= 2,
        f"Ordered ≤ {median_gap:.0f}d ago": c["recency_days"] <= median_gap,
        f"Silent > {p90_gap:.0f}d": c["recency_days"] > p90_gap,
        "Returned goods": c["returned_value"] > 0,
        "Outside UK": c["country"] != "United Kingdom",
    }, index=c.index)


def fig_upset(customers: pd.DataFrame, median_gap: float, p90_gap: float, top: int = 16):
    flags = behaviour_flags(customers, median_gap, p90_gap)
    sets = list(flags.columns)
    key = flags.apply(lambda r: tuple(r.to_numpy()), axis=1)
    rev = customers.loc[flags.index, "monetary"]
    inter = pd.DataFrame({"key": key, "rev": rev}).groupby("key").agg(n=("rev", "size"), rev=("rev", "sum"))
    inter = inter.sort_values("n", ascending=False).head(top)
    inter["rev_share"] = inter["rev"] / rev.sum()

    fig = plt.figure(figsize=(14, 7.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 4.2], height_ratios=[2.2, 1.1, 2.0], wspace=0.45, hspace=0.08)
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_rev = fig.add_subplot(gs[1, 1], sharex=ax_bar)
    ax_mat = fig.add_subplot(gs[2, 1], sharex=ax_bar)
    ax_set = fig.add_subplot(gs[2, 0], sharey=ax_mat)
    x = np.arange(len(inter))

    ax_bar.bar(x, inter["n"], color=CATEGORICAL[0], width=0.62)
    for xi, v in zip(x, inter["n"]):
        ax_bar.text(xi, v + 12, f"{v:,}", ha="center", fontsize=7.8, color=INK)
    ax_bar.set_ylabel("Customers")
    ax_bar.set_title("Exact combinations of behaviour flags (16 largest)", loc="left")
    ax_rev.bar(x, inter["rev_share"], color=INK_2, width=0.62)
    for xi, v in zip(x, inter["rev_share"]):
        if v >= 0.01:
            ax_rev.text(xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=7.6, color=INK)
    ax_rev.set_ylabel("Share of\nnet revenue")
    ax_rev.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    for ax in (ax_bar, ax_rev):
        ax.grid(axis="x", visible=False)
        ax.tick_params(labelbottom=False)

    ys = np.arange(len(sets))[::-1]
    for xi, k in zip(x, inter.index):
        on = [ys[i] for i, v in enumerate(k) if v]
        ax_mat.scatter([xi] * len(sets), ys, s=58, color=GRID, zorder=1)
        if on:
            ax_mat.plot([xi, xi], [min(on), max(on)], color=INK, lw=2, zorder=2)
            ax_mat.scatter([xi] * len(on), on, s=58, color=INK, zorder=3)
    for yi in ys[::2]:
        ax_mat.axhspan(yi - 0.5, yi + 0.5, color=NEUTRAL, zorder=0)
    ax_mat.set_yticks(ys, sets)
    ax_mat.tick_params(axis="y", labelleft=True, labelsize=8.8, labelcolor=INK, pad=4)
    ax_mat.set_xlim(-0.6, len(inter) - 0.4)
    ax_mat.set_xticks([])
    ax_mat.grid(False)
    for s in ax_mat.spines.values():
        s.set_visible(False)
    totals = flags.sum()
    ax_set.barh(ys, totals.to_numpy(), color=CATEGORICAL[0], height=0.55)
    ax_set.invert_xaxis()
    for yi, v, name in zip(ys, totals.to_numpy(), sets):
        ax_set.text(v + 60, yi, f"{v:,}", va="center", ha="right", fontsize=7.8, color=INK)
    ax_set.tick_params(axis="y", labelleft=False, labelright=False)
    ax_set.set_xlabel("Set size")
    ax_set.grid(axis="y", visible=False)
    ax_set.spines["left"].set_visible(False)
    ax_set.spines["right"].set_visible(False)
    best = inter.iloc[int(np.argmax(inter["rev_share"].to_numpy()))]
    headline(fig, f"Returns go with engagement, not exit: {int(best['n']):,} recent repeat top spenders who also return goods "
                  f"hold {best['rev_share']:.0%} of revenue",
             "UpSet plot of six behaviour flags. Each column is an exact combination (dots = flags present).")
    source_note(fig)
    return fig


# 11 -------------------------------------------------------------------------
def fig_repurchase_ridgeline(gaps: pd.DataFrame, median_gap: float, p90_gap: float):
    rows = [("All repeat purchases", gaps["gap_days"], MUTED)]
    rows += [(s, gaps.loc[gaps["segment"] == s, "gap_days"], SEGMENT_COLORS[s]) for s in SEGMENT_ORDER]
    grid = np.linspace(0, np.log10(380), 400)
    fig, ax = plt.subplots(figsize=(11, 6.8))
    step = 1.0
    for i, (name, g, color) in enumerate(rows):
        base = (len(rows) - 1 - i) * step
        v = np.log10(g.to_numpy())
        dens = gaussian_kde(v, bw_method=0.22)(grid)
        dens = dens / dens.max() * 1.25
        ax.fill_between(grid, base, base + dens, color=color, alpha=0.78, lw=0, zorder=10 + i)
        ax.plot(grid, base + dens, color=SURFACE, lw=1.6, zorder=10 + i)
        ax.axhline(base, color=AXIS, lw=0.8, zorder=10 + i)
        med = np.median(g)
        ax.plot([np.log10(med)] * 2, [base, base + dens[np.abs(grid - np.log10(med)).argmin()]], color=INK, lw=1.4,
                zorder=10 + i)
        ax.text(-0.04, base + 0.18, name, ha="right", va="bottom", fontsize=9.5, color=INK, fontweight="medium")
        ax.text(np.log10(380) + 0.03, base + 0.18, f"median {med:.0f} d · {len(g):,} gaps", ha="left", va="bottom",
                fontsize=8.2, color=INK_2)
    for cut, label in ((median_gap, f"{median_gap:.0f} d: 'active'"), (p90_gap, f"{p90_gap:.0f} d: 'lapsed' (P90)")):
        ax.axvline(np.log10(cut), color=INK_2, lw=0.9, zorder=5)
        ax.text(np.log10(cut), len(rows) * step + 0.45, label, fontsize=8.2, color=INK_2, ha="center",
                bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none"), zorder=40)
    _log_axis(ax, "x", 1, 365)
    ax.set_xlim(0, np.log10(380))
    ax.set_ylim(-0.1, len(rows) * step + 0.75)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Days between consecutive purchase days (log scale)")
    headline(fig, "Champions reorder every two weeks; At Risk customers had a two-month rhythm before going quiet",
             "Ridgeline of inter-purchase gaps by segment. The pooled P90 gap (111 d) defines 'lapsed'; pooling weights frequent buyers more.")
    source_note(fig)
    return fig


# 12 -------------------------------------------------------------------------
def fig_segment_value(customers: pd.DataFrame):
    c = _labelled(customers)
    g = c.groupby("segment", observed=True)
    shares = pd.DataFrame({
        "Share of customers": g.size() / len(c),
        "Share of net revenue (past 12 m)": g["monetary"].sum() / c["monetary"].sum(),
        "Share of predicted revenue (next 180 d)": g["clv"].sum() / c["clv"].sum(),
    })
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True, gridspec_kw={"wspace": 0.12})
    y = np.arange(len(shares))[::-1]
    for ax, col in zip(axes, shares.columns):
        ax.barh(y, shares[col], color=[SEGMENT_COLORS[s] for s in shares.index], height=0.6)
        for yi, v in zip(y, shares[col]):
            ax.text(v + 0.01, yi, f"{v:.0%}", va="center", fontsize=8.8, color=INK)
        ax.set_title(col)
        ax.set_xlim(0, 0.62)
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, shares.index)
    ch = shares.loc["Champions"]
    headline(fig, f"{ch.iloc[0]:.0%} of customers (Champions) produced {ch.iloc[1]:.0%} of revenue and are forecast to "
                  f"produce {ch.iloc[2]:.0%} of the next six months'",
             "Predicted revenue = BG/NBD expected purchase days x Gamma-Gamma expected spend; it assumes no churn (see CLV caveats).")
    source_note(fig)
    return fig, shares


# 13 -------------------------------------------------------------------------
def fig_clv_validation(holdout: pd.DataFrame, tracking: pd.DataFrame, metrics: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw={"wspace": 0.3, "width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    days = np.arange(1, len(tracking) + 1)
    ax.plot(days, tracking["actual_cum"], color=INK, label="Actual")
    ax.plot(days, tracking["bgnbd_cum"], color=CATEGORICAL[0], label="BG/NBD forecast")
    ax.plot(days, tracking["repeat_rate_cum"], color=CATEGORICAL[1], label="Repeat-rate baseline")
    ax.text(days[-1] + 1, tracking["actual_cum"].iloc[-1], f"Actual {tracking['actual_cum'].iloc[-1]:,.0f}", fontsize=8.3, va="center")
    ax.text(days[-1] + 1, tracking["bgnbd_cum"].iloc[-1] + 60, f"BG/NBD {tracking['bgnbd_cum'].iloc[-1]:,.0f}", fontsize=8.3,
            va="center", color=INK)
    ax.text(days[-1] + 1, tracking["repeat_rate_cum"].iloc[-1] - 60, f"Baseline {tracking['repeat_rate_cum'].iloc[-1]:,.0f}",
            fontsize=8.3, va="center", color=INK)
    ax.set_xlim(0, days[-1] + 22)
    ax.set_xlabel("Days into hold-out (10 Sep - 9 Dec 2011)")
    ax.set_ylabel("Cumulative purchase days")
    ax.set_title("a  Aggregate tracking: both forecasts miss the Q4 surge")
    ax.legend(loc="upper left")

    h = holdout.copy()
    h["freq_bin"] = h["x"].clip(upper=7)
    h["rec_bin"] = pd.cut(h["T"] - h["t_x"], [-1, 14, 30, 60, 90, 120, 180, 270, 400],
                          labels=["≤14", "15-30", "31-60", "61-90", "91-120", "121-180", "181-270", "271+"])
    for ax, col, title, xlabel in ((axes[1], "freq_bin", "b  By calibration-period repeat purchases", "Repeat purchase days before 10 Sep (7 = 7+)"),
                                   (axes[2], "rec_bin", "c  By days silent at 9 Sep", "Days since last purchase at calibration end")):
        m = h.groupby(col, observed=True)[["actual", "bgnbd", "repeat_rate"]].mean()
        xx = np.arange(len(m))
        ax.plot(xx, m["actual"], color=INK, marker="o", ms=6, mec=SURFACE, mew=1.5, label="Actual")
        ax.plot(xx, m["bgnbd"], color=CATEGORICAL[0], marker="o", ms=6, mec=SURFACE, mew=1.5, label="BG/NBD")
        ax.plot(xx, m["repeat_rate"], color=CATEGORICAL[1], marker="o", ms=6, mec=SURFACE, mew=1.5, label="Repeat-rate baseline")
        ax.set_xticks(xx, m.index.astype(str), fontsize=7.8, rotation=35 if col == "rec_bin" else 0,
                      ha="right" if col == "rec_bin" else "center", rotation_mode="anchor")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean purchase days in hold-out")
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="upper left")
    p = metrics
    headline(fig, f"BG/NBD is no more accurate than a simple repeat-rate baseline (MAE {p['holdout_bgnbd_mae']:.2f} vs "
                  f"{p['holdout_repeat_rate_mae']:.2f}); both miss the Q4 surge by "
                  f"{1 - p['holdout_bgnbd_total'] / p['holdout_actual_total']:.0%}-{1 - p['holdout_repeat_rate_total'] / p['holdout_actual_total']:.0%}",
             f"Fit on 1 Dec 2010 - 9 Sep 2011, scored on the next 13 weeks (n = {len(holdout):,}). Baseline = repeat purchase days / days of tenure.\n"
             f"Spearman ρ: BG/NBD {p['holdout_bgnbd_spearman']:.2f}, baseline {p['holdout_repeat_rate_spearman']:.2f}. "
             "Customers silent 270+ days still bought in Q4 (n = 143): long gaps here are partly seasonal.")
    source_note(fig)
    return fig


# 14 -------------------------------------------------------------------------
def _cohort_label(p: pd.Period) -> str:
    if p == pd.Period("2010-12", "M"):
        return "Dec 2010 (incl. legacy)"
    return f"2011 Q{p.quarter} acquired"


def fig_lifecycle_sankey(customers: pd.DataFrame):
    c = _labelled(customers)
    c["stage0"] = c["cohort_month"].apply(_cohort_label)
    c["stage1"] = c["segment"].astype(str)
    c["stage2"] = c["clv_tier"].astype(str)
    stages = {
        "stage0": ["Dec 2010 (incl. legacy)", "2011 Q1 acquired", "2011 Q2 acquired", "2011 Q3 acquired", "2011 Q4 acquired"],
        "stage1": SEGMENT_ORDER,
        "stage2": ["Top 20% CLV", "Middle 40%", "Bottom 40%"],
    }
    xs = {"stage0": 0.0, "stage1": 1.0, "stage2": 2.0}
    total, gap, bar_w = len(c), 0.035, 0.035
    span = 1.0

    nodes = {}
    for st, names in stages.items():
        sizes = c[st].value_counts().reindex(names, fill_value=0)
        free = span - gap * (len(names) - 1)
        y = 0.0
        for name in names:
            hgt = sizes[name] / total * free
            nodes[(st, name)] = [y, hgt, y, y]  # top, height, out-cursor, in-cursor
            y += hgt + gap

    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(-0.55, 2.6)
    ax.set_ylim(span + 0.02, -0.04)
    ax.axis("off")

    def band(x0, y0, h0, x1, y1, h1, color):
        xm = (x0 + x1) / 2
        verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 + h1), (xm, y1 + h1), (xm, y0 + h0), (x0, y0 + h0), (x0, y0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        ax.add_patch(PathPatch(MplPath(verts, codes), fc=color, ec="none", alpha=0.42))

    for left, right in (("stage0", "stage1"), ("stage1", "stage2")):
        flows = c.groupby([left, right]).size()
        for ln in stages[left]:
            for rn in stages[right]:
                n = flows.get((ln, rn), 0)
                if n == 0:
                    continue
                hgt = n / total * (span - gap * (len(stages[left]) - 1))
                hgt_r = n / total * (span - gap * (len(stages[right]) - 1))
                src, dst = nodes[(left, ln)], nodes[(right, rn)]
                seg = rn if left == "stage0" else ln
                band(xs[left] + bar_w / 2, src[2], hgt, xs[right] - bar_w / 2, dst[3], hgt_r, SEGMENT_COLORS[seg])
                src[2] += hgt
                dst[3] += hgt_r

    for (st, name), (top, hgt, *_rest) in nodes.items():
        color = SEGMENT_COLORS.get(name, INK_2) if st == "stage1" else INK_2
        ax.add_patch(Rectangle((xs[st] - bar_w / 2, top), bar_w, hgt, color=color, lw=0))
        n = int((c[st] == name).sum())
        if st == "stage0":
            ax.text(xs[st] - bar_w, top + hgt / 2, f"{name}\n{n:,}", ha="right", va="center", fontsize=8.8, color=INK)
        elif st == "stage2":
            ax.text(xs[st] + bar_w, top + hgt / 2, f"{name}\n{n:,}", ha="left", va="center", fontsize=8.8, color=INK)
        else:
            ax.text(xs[st] + bar_w, top + hgt / 2, f"{name}  {n:,}", ha="left", va="center", fontsize=8.8,
                    color=INK, fontweight="semibold", bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))
    for st, label in (("stage0", "When acquired"), ("stage1", "Value segment (K-means)"), ("stage2", "Predicted 180-day CLV")):
        ax.text(xs[st], -0.03, label, ha="center", va="bottom", fontsize=9.5, color=INK_2, fontweight="semibold")
    legacy_share = (c.loc[c["stage1"] == "Champions", "stage0"] == "Dec 2010 (incl. legacy)").mean()
    top = c[c["stage2"] == "Top 20% CLV"]
    top_share = top["stage1"].isin(["Champions", "Loyal"]).mean()
    headline(fig, f"{legacy_share:.0%} of Champions were already buying in the first month of data; "
                  f"{top_share:.0%} of the top CLV tier are Champions or Loyal",
             "Alluvial diagram: acquisition period → K-means segment → predicted CLV tier. Band width = customers; colour = segment.")
    source_note(fig)
    return fig


# ----------------------------------------------------------------------------
def make_all_figures(res) -> list:
    m = res.metrics
    paths = [save(fig_data_audit(res.audit, res.tutorial, res.customers, m), "01_data_audit")]
    fig, _ = fig_revenue_concentration(res.customers, m)
    paths.append(save(fig, "02_revenue_concentration"))
    paths.append(save(fig_monthly_activity(res.monthly), "03_monthly_activity"))
    paths.append(save(fig_cohort_heatmap(res.retention, res.cohort_sizes), "04_cohort_retention_heatmap"))
    paths.append(save(fig_k_selection(res.k_selection), "05_k_selection"))
    paths.append(save(fig_segment_violins(res.customers), "06_segment_violins"))
    paths.append(save(fig_segment_profile(res.customers), "07_segment_profile_heatmap"))
    paths.append(save(fig_segment_tree(res.tree, res.tree_fidelity), "08_segment_decision_tree"))
    paths.append(save(fig_kmeans_vs_rules(res.customers, m["kmeans_vs_rfm_ami"]), "09_kmeans_vs_rfm_rules"))
    paths.append(save(fig_upset(res.customers, m["median_gap_days"], m["p90_gap_days"]), "10_behaviour_upset"))
    paths.append(save(fig_repurchase_ridgeline(res.gaps[res.gaps["segment"].isin(SEGMENT_ORDER)],
                                               m["median_gap_days"], m["p90_gap_days"]), "11_repurchase_ridgeline"))
    fig, _ = fig_segment_value(res.customers)
    paths.append(save(fig, "12_segment_value_shares"))
    paths.append(save(fig_clv_validation(res.holdout, res.holdout_tracking, m), "13_clv_validation"))
    paths.append(save(fig_lifecycle_sankey(res.customers), "14_lifecycle_sankey"))
    return paths
