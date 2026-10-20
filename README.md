# 客户行为分析：RFM · K-means 分群 · 队列留存 · 概率型 CLV

> Customer behaviour analytics on the UCI Online Retail dataset — SQL + Python RFM features, K-means segmentation with stability checks, cohort retention, repeat-purchase cycles and a validated BG/NBD + Gamma-Gamma CLV model, plus an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/python-3.10%2B-2a78d6) [![tests](https://github.com/rollawang02-cyber/customer-rfm-analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/rollawang02-cyber/customer-rfm-analytics/actions/workflows/tests.yml) ![SQL](https://img.shields.io/badge/SQL-DuckDB-eda100) ![license](https://img.shields.io/badge/license-MIT-52514e)

用 SQL 和 Python 分析一家英国在线礼品批发商 13 个月的交易明细（541,909 行、4,334 位注册客户），回答四个问题：**谁是高价值客户、他们多久复购一次、谁在流失、未来半年的收入从哪里来。**
项目从一个公开的 RFM 教程出发，先审计并修正了它的几处错误，再在此基础上做分群、留存和预测。

---

## 核心结论

| | 发现 | 证据 |
|---|---|---|
| 1 | **收入极度集中**：前 20% 客户贡献 74% 净收入，前 1% 贡献 30%（Gini 0.71）；35% 的客户只买过一次 | 图 2 |
| 2 | **5 个分群**：Champions 占 8% 的客户却贡献 53% 的收入；At Risk（1,000 人、16% 收入）曾有约 60 天的复购节奏，现已沉寂 84 天（中位） | 图 6、7、12 |
| 3 | **增长来自老客户回流**：9–11 月净收入比 6–8 月高 63%，主要靠回头客；70% 的 Champions 在数据的第一个月就已在下单（多为左删失的老客户） | 图 3、14 |
| 4 | **复购周期**：所有复购间隔的中位数 28 天、P90 111 天；Champions 每 14 天复购一次 | 图 11 |
| 5 | **长时间沉默很大程度上是季节性的**：到 9 月已沉寂 270 天以上的客户，有 38% 在 Q4 又回来下单（n = 143）；BG/NBD 也几乎识别不到流失过程 | 图 13c |
| 6 | **CLV 模型要诚实地评估**：BG/NBD 在留出期的精度与简单的复购速率基线相当（MAE 0.93 vs 0.90，Spearman 均为 0.50），只明显优于把首单也计入的 run-rate 基线（MAE 1.63）；两者都低估了旺季，分别低 26% 和 30% | 图 13 |
| 7 | **退货往往和高参与度一起出现**：379 位"消费前 20% + 复购 + 近期活跃 + 有退货"的客户贡献了 37% 的收入 | 图 10 |

## 对参考教程的批判与修正

参考教程（一份公开的 Python RFM 客户分层 notebook）的流程被完整复现（`rfm.features.replicate_tutorial_rfm`，输出与教程打印值 `296.7611` 完全一致），并逐项修正：

| 问题 | 影响 | 本项目的做法 |
|---|---|---|
| `df_f = df` 后 `drop_duplicates(..., inplace=True)`：别名导致**原表被去重为每张发票只剩第一行** | Monetary 只捕获 **14.7%** 的真实收入（均值 £297，正确值 £1,908） | 在完整明细上求和；测试用手算样例锁定 |
| 直接删除退货（Quantity < 0），不做抵扣 | 客户 12346 下单 £77k、16 分钟后全退，被当成头部客户 | 保留退货为负收入，**Monetary = 购买 − 退货** |
| 邮费、Amazon 手续费、手工调整、坏账、礼品券被当成消费 | 金额被系统性抬高或压低 | 显式的非商品代码清单（`config.NON_MERCH_CODES`） |
| Recency 以 2012-01-01 为基准 | 多数客户凭空多出 21 天 | 以最后发票日次日 2011-12-10 为快照日 |
| 8 类分群用**均值**作阈值 | 偏度 > 20 的分布上均值没有代表性 | 对数变换 + K-means；另做规则型 RFM（R、M 五分位 + F 固定分箱）作对照 |
| K-means 在**原始尺度**上运行；`k=8` 模型拟合后未被使用 | 距离几乎只由 Monetary 决定 | `log1p` + 标准化；用 4 个指标 + bootstrap 稳定性选 k |
| 没有验证、没有解释 | 分群无法落地 | 代理决策树把簇翻译成"天 / 单数 / 英镑"规则（保真度 84%） |

其他数据层面的发现：
- **25% 的行没有 CustomerID**（游客结账），占约 15% 的收入——所有客户级结论只覆盖注册客户。
- **2010-12"队列"是左删失的**：数据从 2010-12-01 开始，所有老客户都被标成"12 月新客"，所以它的留存率异常高（图 4）。
- **2011-12 只有 9 天**：按月的图如果不标注，会被误读为断崖式下跌（图 3）。
- 同一天下单后全部取消的客户-日不计为交易，否则客户 16446 那笔 80,995 件、£168k 的即退订单会给他一个六位数的 CLV。

## 方法

```
raw CSV ─► 清洗规则 (data.py / 01_clean_transactions.sql)
        ─► 客户级 RFM (features.py / 02_customer_rfm.sql)   ── SQL 与 pandas 逐列比对，最大差异 < 1e-9
        ─► 月度活跃 / 队列留存 / 复购间隔 (retention.py / 03_cohort_retention.sql)
        ─► log1p + z-score ─► K-means k=2..8 ─► Silhouette · DB · CH · bootstrap ARI ─► k = 5
                                             ─► 代理决策树（可解释规则）  ─► 与 R×F 规则分群对照 (AMI)
        ─► BG/NBD + Gamma-Gamma（按论文闭式似然自行实现）
             ├─ 校准期 2010-12-01 ~ 2011-09-09 ─► 留出 13 周验证 vs 复购速率 / run-rate 基线
             └─ 全量拟合 ─► 180 天预测收入（CLV）
```

**选 k 的诚实说明**：k ≥ 3 时 Silhouette 只有 0.30–0.34，说明 RFM 空间是连续体而不是天然分离的簇。分群是为了行动而做的切分。k = 5 在 k ≥ 4 中 Davies-Bouldin 最低，平均 bootstrap ARI（0.88）与 k = 4 相当但波动更大；选它主要是因为可操作性：它是唯一单独分出 "At Risk" 的解（图 5）。

**流失定义**：BG/NBD 给出的平均流失概率接近 0，所有客户的 P(alive) 都 > 0.95，因此不用它当流失标签。改用复购间隔的经验分布：Recency ≤ 28 天（中位间隔）为 Active，28–111 天为 Cooling，> 111 天（P90）为 Lapsed。这两个分位数是把所有复购间隔合并计算的，高频客户贡献的间隔更多；如果按客户先取中位数，再算分位数，结果约为 52 / 166 天，所以对低频客户来说，111 天这个阈值偏严格。

## 图表

14 张图都嵌在 [notebook](notebooks/customer_rfm_analysis.ipynb) 里，GitHub 会直接显示。本地运行 `python -m rfm.pipeline` 会把它们以 PNG 格式生成到 `reports/figures/`（这个文件夹不上传）。

| # | 图 | 回答的问题 |
|---|---|---|
| 1 | 数据审计 + 教程 bug 散点 | 清洗删掉了什么？教程的 Monetary 错在哪？ |
| 2 | Lorenz 曲线 | 收入有多集中？ |
| 3 | 月度新老客户与收入 | 增长来自新客还是老客？ |
| 4 | 队列留存热图 | 各月获客的留存如何？左删失的影响 |
| 5 | 选 k：4 个指标 + bootstrap 稳定性 | 为什么是 5 群？ |
| 6 | 分群 RFM 小提琴图 | 各群的 R / F / M 分布 |
| 7 | 分群画像热图 | 各群在 9 个指标上的差异 |
| 8 | 代理决策树 | 怎样用业务规则复现聚类？ |
| 9 | K-means vs 规则型 RFM 热图 | 两种分群方法有多一致？ |
| 10 | 行为交集（UpSet） | 哪些行为组合贡献收入？ |
| 11 | 复购间隔脊线图 | 各群多久复购一次？ |
| 12 | 人数 / 历史收入 / 预测收入 | 价值集中在哪些群？ |
| 13 | CLV 留出期验证 | BG/NBD 比基线好吗？ |
| 14 | 生命周期桑基图 | 获客时期 → 分群 → CLV 档位 |

配色使用一套经过色觉障碍（CVD）校验的分类色板：相邻色对在红绿色盲模拟下 ΔE ≥ 9；每张多系列图都带图例或直接标注，不单靠颜色区分。顺序数据用单色系蓝，正负偏离用蓝-灰-红发散色板。

## 交互式看板

```bash
streamlit run dashboard/app.py
```

侧栏按分群 / 地区 / 活跃状态筛选；4 个页签：概览（KPI、月度活跃、人数 vs 价值）、分群（Recency × Monetary 散点、分布箱线图、画像表）、留存（队列热图）、客户（可搜索、可下载 CSV）。看板只读取 `data/processed/` 下的结果表，不重新计算。

## 快速开始

```bash
git clone https://github.com/rollawang02-cyber/customer-rfm-analytics.git
cd customer-rfm-analytics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
# 如需逐位复现 README 中的数字：pip install -r requirements-lock.txt（在 Python 3.13 上测试过）

python -m rfm.pipeline          # 重算所有表和图（约 25 秒）
pytest                          # 10 个测试
jupyter lab notebooks/customer_rfm_analysis.ipynb
streamlit run dashboard/app.py
```


## 项目结构

```
├── data/
│   ├── raw/online_retail.zip        # UCI Online Retail 原始数据（CC BY 4.0）
│   └── processed/                   # pipeline 输出：customers.csv、cohort_retention.csv、metrics.json …
├── sql/                             # DuckDB 实现：清洗、RFM、队列留存
├── src/rfm/
│   ├── config.py                    # 所有会改变结论的假设（快照日、非商品代码、验证切分）
│   ├── data.py                      # 读取 + 有名字的清洗步骤 + 审计表
│   ├── features.py                  # 客户级 RFM；教程复现
│   ├── sql.py                       # 运行 sql/ 并与 pandas 对齐
│   ├── retention.py                 # 队列、复购间隔、月度活跃、流失阈值
│   ├── segmentation.py              # 缩放、选 k、K-means、规则 RFM、代理树
│   ├── clv.py                       # BG/NBD 与 Gamma-Gamma（闭式似然）
│   ├── pipeline.py                  # 端到端：python -m rfm.pipeline
│   └── viz/                         # 统一主题与 14 张图
├── notebooks/customer_rfm_analysis.ipynb   # 叙述层（中文），已执行
├── dashboard/app.py                 # Streamlit 看板
├── reports/figures/                 # 运行后生成的 PNG（已在 .gitignore 中，不上传）
└── tests/                           # pytest：清洗、RFM 手算样例、SQL 一致性、队列、CLV 参数回收
```

## 测试

测试只放在公共接口上，期望值来自手算样例或已知参数的模拟数据，而不是重新实现一遍被测逻辑：

| 测试 | 验证什么 |
|---|---|
| `test_data.py` | 清洗只留下"可识别客户的商品行"；退货以负收入保留 |
| `test_features.py` | 一个手算的 3 客户样例：R、F、M（净额）、在网时长、退货率 |
| `test_sql_parity.py` | SQL 管道与 pandas 的 RFM 完全一致 |
| `test_retention.py` | 队列留存矩阵、SQL 队列与 pandas 一致、复购间隔用"不同购买日" |
| `test_clv.py` | BG/NBD 与 Gamma-Gamma 能从模拟数据中回收已知参数；BG/NBD 输入构造正确 |

每次 push 时，GitHub Actions（`.github/workflows/tests.yml`）都会自动运行测试。

## 局限

- 只覆盖有 CustomerID 的交易（约 85% 收入）。
- 13 个月只含一个旺季，季节性与趋势无法分离；CLV 只做 180 天、未折现、收入口径（无毛利数据）。
- 约 90% 的客户在英国，海外结论样本量小。
- Gamma-Gamma 假设频率与金额独立，这里两者相关系数为 0.17，属于轻度违反。
- 分群边界附近的客户不稳定（Silhouette ≈ 0.32），落地时建议配合代理决策树的规则使用。

## 数据与引用

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 — CC BY 4.0.

Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). "Counting your customers" the easy way: An alternative to the Pareto/NBD model. *Marketing Science*, 24(2), 275–284.

Fader, P. S., Hardie, B. G. S., & Lee, K. L. (2005). RFM and CLV: Using iso-value curves for customer base analysis. *Journal of Marketing Research*, 42(4), 415–430.

## License

代码使用 MIT License；数据遵循其原始的 CC BY 4.0 许可。
