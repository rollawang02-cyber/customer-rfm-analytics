"""Interactive customer analytics dashboard.

Run from the repository root:  streamlit run dashboard/app.py
Reads the tables written by `python -m rfm.pipeline` (data/processed/).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data" / "processed"
SEGMENTS = ["Champions", "Loyal", "New / Promising", "At Risk", "Hibernating"]
COLORS = dict(zip(SEGMENTS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]

st.set_page_config(page_title="Customer RFM Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def load():
    customers = pd.read_csv(DATA / "customers.csv", dtype={"CustomerID": str}, parse_dates=["first_purchase", "last_purchase"])
    customers = customers[customers["segment"].isin(SEGMENTS)]
    monthly = pd.read_csv(DATA / "monthly_activity.csv")
    retention = pd.read_csv(DATA / "cohort_retention.csv", index_col=0)
    metrics = json.loads((DATA / "metrics.json").read_text())
    return customers, monthly, retention, metrics


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK_2),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0, title=None), hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


customers, monthly, retention, metrics = load()

# ---- filters (one row, in the sidebar) ------------------------------------
st.sidebar.header("筛选 Filters")
seg_pick = st.sidebar.multiselect("分群 Segment", SEGMENTS, default=SEGMENTS)
region = st.sidebar.radio("地区 Region", ["全部 All", "英国 UK", "海外 Outside UK"], horizontal=False)
statuses = sorted(customers["status"].unique(), key=lambda s: ["Active", "Cooling", "Lapsed"].index(s.split()[0]))
status_pick = st.sidebar.multiselect("活跃状态 Status", statuses, default=statuses)

view = customers[customers["segment"].isin(seg_pick) & customers["status"].isin(status_pick)]
if region.startswith("英国"):
    view = view[view["country"] == "United Kingdom"]
elif region.startswith("海外"):
    view = view[view["country"] != "United Kingdom"]

st.title("客户行为分析看板 · Customer Behaviour Dashboard")
st.caption(
    "UCI Online Retail, 2010-12-01 → 2011-12-09 · identified customers only · "
    f"K-means (k=5) on log-scaled RFM · CLV = BG/NBD × Gamma-Gamma, {180}-day horizon"
)

if view.empty:
    st.warning("当前筛选没有客户。No customers match the current filters.")
    st.stop()

# ---- KPI tiles --------------------------------------------------------------
k = st.columns(5)
k[0].metric("客户数 Customers", f"{len(view):,}", help=f"占全部客户 {len(view) / len(customers):.0%}")
k[1].metric("净收入 Net revenue", f"£{view['monetary'].sum() / 1e6:,.2f}M",
            help=f"占全部净收入 {view['monetary'].sum() / customers['monetary'].sum():.0%}")
k[2].metric("复购率 Repeat rate", f"{(view['frequency'] >= 2).mean():.0%}")
k[3].metric("中位 Recency", f"{view['recency_days'].median():.0f} 天")
k[4].metric("预测 180 天收入", f"£{view['clv'].sum() / 1e6:,.2f}M")

tab1, tab2, tab3, tab4 = st.tabs(["概览 Overview", "分群 Segments", "留存 Retention", "客户 Customers"])

with tab1:
    c1, c2 = st.columns([1.4, 1])
    with c1:
        mm = monthly.copy()
        fig = go.Figure()
        fig.add_bar(x=mm["order_month"], y=mm["returning"], name="Returning", marker_color="#2a78d6",
                    hovertemplate="%{x}<br>Returning: %{y:,}<extra></extra>")
        fig.add_bar(x=mm["order_month"], y=mm["new"], name="New", marker_color="#1baf7a",
                    hovertemplate="%{x}<br>New: %{y:,}<extra></extra>")
        fig.update_layout(barmode="stack", title="月度活跃客户（全体）· 2011-12 仅含 1–9 日")
        st.plotly_chart(style(fig), width="stretch")
    with c2:
        g = view.groupby("segment").agg(customers=("frequency", "size"), revenue=("monetary", "sum"), clv=("clv", "sum"))
        g = g.reindex([s for s in SEGMENTS if s in g.index])
        long = pd.DataFrame({
            "segment": np.repeat(g.index, 3),
            "measure": ["客户占比", "历史收入占比", "预测收入占比"] * len(g),
            "share": np.column_stack([g["customers"] / g["customers"].sum(), g["revenue"] / g["revenue"].sum(),
                                      g["clv"] / g["clv"].sum()]).ravel(),
        })
        fig = px.bar(long, x="share", y="measure", color="segment", orientation="h", color_discrete_map=COLORS,
                     category_orders={"segment": SEGMENTS})
        fig.update_traces(hovertemplate="%{fullData.name}: %{x:.1%}<extra></extra>")
        fig.update_layout(title="分群的人数 vs 价值", xaxis_tickformat=".0%", yaxis_title=None, xaxis_title=None)
        st.plotly_chart(style(fig), width="stretch")

    st.subheader("关键发现")
    st.markdown(
        f"""
- 前 20% 客户贡献 **{metrics['top20_revenue_share']:.0%}** 的净收入；{metrics['one_time_buyer_share']:.0%} 的客户只买过一次。
- 复购间隔中位数 **{metrics['median_gap_days']:.0f} 天**，P90 为 **{metrics['p90_gap_days']:.0f} 天**（用作"沉寂"阈值）。
- BG/NBD 留出期 MAE {metrics['holdout_bgnbd_mae']:.2f}，与简单复购速率基线（{metrics['holdout_repeat_rate_mae']:.2f}）相当；
  两者都低估了 Q4 旺季（BG/NBD 低估 {1 - metrics['holdout_bgnbd_total'] / metrics['holdout_actual_total']:.0%}）。
- 参考教程的 Monetary 由于别名 + inplace 去重 bug，只捕获了 {metrics['tutorial_revenue_captured']:.1%} 的收入。
"""
    )

with tab2:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        sample = view.sample(min(len(view), 3000), random_state=0)
        fig = px.scatter(sample, x="recency_days", y="monetary", color="segment", log_x=True, log_y=True,
                         color_discrete_map=COLORS, category_orders={"segment": SEGMENTS},
                         hover_data={"CustomerID": True, "frequency": True, "clv": ":,.0f", "country": True})
        fig.update_traces(marker=dict(size=7, opacity=0.7, line=dict(width=1, color="white")))
        fig.update_layout(title="Recency × Monetary（对数轴，最多抽样 3,000 人）", xaxis_title="Recency (days)",
                          yaxis_title="Net spend (£)")
        fig.update_xaxes(dtick=1)
        fig.update_yaxes(dtick=1)
        st.plotly_chart(style(fig, 460), width="stretch")
    with c2:
        metric = st.selectbox("分布指标", ["frequency", "monetary", "recency_days", "aov", "clv"], index=1)
        fig = px.box(view[view[metric] > 0], x="segment", y=metric, color="segment", points=False, log_y=True,
                     color_discrete_map=COLORS, category_orders={"segment": SEGMENTS})
        fig.update_yaxes(dtick=1)
        fig.update_layout(showlegend=False, title=f"{metric} 分布（箱线图，对数轴）", xaxis_title=None)
        st.plotly_chart(style(fig, 460), width="stretch")
    prof = view.groupby("segment").agg(
        客户数=("frequency", "size"), 中位Recency=("recency_days", "median"), 中位订单数=("frequency", "median"),
        中位净消费=("monetary", "median"), 中位AOV=("aov", "median"), 中位CLV=("clv", "median"),
    ).reindex([s for s in SEGMENTS if s in seg_pick])
    st.dataframe(prof.style.format({"中位净消费": "£{:,.0f}", "中位AOV": "£{:,.0f}", "中位CLV": "£{:,.0f}",
                                    "中位Recency": "{:.0f}", "中位订单数": "{:.0f}"}), width="stretch")

with tab3:
    ret = retention.drop(columns="0").iloc[:-1, :12]
    z = ret.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"+{c}" for c in ret.columns], y=ret.index, colorscale=[[i / 8, c] for i, c in enumerate(BLUES)],
        zmin=0, zmax=0.45, colorbar=dict(tickformat=".0%", title="留存"),
        text=np.where(np.isnan(z), "", np.vectorize(lambda v: f"{v:.0%}")(np.nan_to_num(z))), texttemplate="%{text}",
        hovertemplate="Cohort %{y}<br>Month %{x}<br>Retention %{z:.1%}<extra></extra>", xgap=2, ygap=2,
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title="月度获客队列留存（2010-12 为左删失的存量客户，对角线末格为仅 9 天的 2011-12）",
                      xaxis_title="Months since first purchase")
    st.plotly_chart(style(fig, 520), width="stretch")

with tab4:
    q = st.text_input("按 CustomerID 搜索", "")
    cols = ["CustomerID", "segment", "rfm_segment", "status", "country", "recency_days", "frequency", "monetary",
            "aov", "return_rate", "exp_purchases", "clv", "clv_tier"]
    table = view[cols].sort_values("clv", ascending=False)
    if q:
        table = table[table["CustomerID"].str.contains(q.strip())]
    st.dataframe(table, width="stretch", height=480, hide_index=True,
                 column_config={"monetary": st.column_config.NumberColumn("net spend", format="£%.0f"),
                                "clv": st.column_config.NumberColumn("CLV 180d", format="£%.0f"),
                                "aov": st.column_config.NumberColumn(format="£%.0f"),
                                "return_rate": st.column_config.NumberColumn(format="%.2f"),
                                "exp_purchases": st.column_config.NumberColumn(format="%.1f")})
    st.download_button("下载当前筛选 CSV", table.to_csv(index=False).encode("utf-8"), "customers_filtered.csv", "text/csv")
