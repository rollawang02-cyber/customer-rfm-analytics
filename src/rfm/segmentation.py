"""K-means segmentation on log-scaled RFM, its diagnostics, a rule-based RFM
benchmark and an interpretable surrogate tree."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE

RFM_COLS = ["recency_days", "frequency", "monetary"]
SEGMENT_ORDER = ["Champions", "Loyal", "New / Promising", "At Risk", "Hibernating"]


def clusterable(rfm: pd.DataFrame) -> pd.DataFrame:
    """Customers with positive net spend; net <= 0 means everything was returned."""
    return rfm[rfm["monetary"] > 0]


def make_scaler() -> Pipeline:
    """log1p then z-score. RFM is heavily right-skewed (Monetary skew > 20); on
    raw values Euclidean K-means is driven almost entirely by Monetary."""
    return make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())


def evaluate_k(X: np.ndarray, ks=range(2, 9), n_boot: int = 20) -> pd.DataFrame:
    """Internal validity indices plus bootstrap stability (mean ARI vs. full-data fit)."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for k in ks:
        ref = KMeans(k, n_init=20, random_state=RANDOM_STATE).fit(X)
        aris = []
        for i in range(n_boot):
            idx = rng.choice(len(X), len(X), replace=True)
            boot = KMeans(k, n_init=5, random_state=i).fit(X[idx])
            aris.append(adjusted_rand_score(ref.labels_, boot.predict(X)))
        rows.append(
            {
                "k": k,
                "inertia": ref.inertia_,
                "silhouette": silhouette_score(X, ref.labels_),
                "calinski_harabasz": calinski_harabasz_score(X, ref.labels_),
                "davies_bouldin": davies_bouldin_score(X, ref.labels_),
                "stability_ari": float(np.mean(aris)),
                "stability_ari_p05": float(np.quantile(aris, 0.05)),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def _name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    """Deterministic names for the k=5 solution from median R/F/M per cluster.

    value = rank of median F + rank of median M. Highest value -> Champions.
    Of the two lowest-value (one-off) clusters, the recent one is New / Promising
    and the dormant one Hibernating. Of the remaining two, the more recent is
    Loyal and the less recent At Risk.
    """
    if len(profile) != 5:
        return {c: f"Segment {i + 1}" for i, c in enumerate(profile.index)}
    value = profile["frequency"].rank() + profile["monetary"].rank()
    order = value.sort_values(ascending=False).index.tolist()
    names = {order[0]: "Champions"}
    low = profile.loc[order[-2:]].sort_values("recency_days").index
    names[low[0]], names[low[1]] = "New / Promising", "Hibernating"
    mid = profile.loc[order[1:3]].sort_values("recency_days").index
    names[mid[0]], names[mid[1]] = "Loyal", "At Risk"
    return names


@dataclass
class Segmenter:
    k: int = 5
    scaler: Pipeline | None = None
    model: KMeans | None = None
    names: dict | None = None

    def fit(self, rfm: pd.DataFrame) -> "Segmenter":
        data = clusterable(rfm)[RFM_COLS]
        self.scaler = make_scaler().fit(data)
        self.model = KMeans(self.k, n_init=50, random_state=RANDOM_STATE).fit(self.scaler.transform(data))
        profile = data.groupby(self.model.labels_).median()
        self.names = _name_clusters(profile)
        return self

    def predict(self, rfm: pd.DataFrame) -> pd.Series:
        data = clusterable(rfm)[RFM_COLS]
        labels = self.model.predict(self.scaler.transform(data))
        seg = pd.Series([self.names[c] for c in labels], index=data.index, name="segment")
        order = SEGMENT_ORDER if self.k == 5 else sorted(set(seg))
        return seg.astype(pd.CategoricalDtype(order, ordered=True))


def rfm_scores(rfm: pd.DataFrame) -> pd.DataFrame:
    """Classic 1-5 scores and the widely used R x F segment map.

    R and M are quintiles of the raw values (R reversed: recent = 5). Frequency
    is *not* quintiled: 35% of customers bought exactly once, so any quintile
    split must break ties arbitrarily and would give identical customers
    different scores. Fixed, business-readable bins are used instead:
    1 / 2 / 3-4 / 5-8 / 9+ orders.
    """
    out = pd.DataFrame(index=rfm.index)
    out["R"] = pd.qcut(rfm["recency_days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    out["F"] = pd.cut(rfm["frequency"], [0, 1, 2, 4, 8, np.inf], labels=[1, 2, 3, 4, 5]).astype(int)
    out["M"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    seg_map = {
        r"[1-2][1-2]": "Hibernating",
        r"[1-2][3-4]": "At Risk",
        r"[1-2]5": "Can't Lose",
        r"3[1-2]": "About to Sleep",
        r"33": "Need Attention",
        r"[3-4][4-5]": "Loyal",
        r"41": "Promising",
        r"51": "New",
        r"[4-5][2-3]": "Potential Loyalist",
        r"5[4-5]": "Champions",
    }
    rf = out["R"].astype(str) + out["F"].astype(str)
    out["rfm_segment"] = rf.replace(seg_map, regex=True)
    return out


def surrogate_tree(rfm: pd.DataFrame, segments: pd.Series, depth: int = 3):
    """Shallow tree that re-expresses the K-means partition as business rules.

    Returns the fitted tree and its 5-fold cross-validated fidelity (accuracy
    at reproducing the cluster labels from raw R, F, M).
    """
    X = rfm.loc[segments.index, RFM_COLS]
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=40, random_state=RANDOM_STATE)
    fidelity = cross_val_score(tree, X, segments.astype(str), cv=5).mean()
    return tree.fit(X, segments.astype(str)), float(fidelity)
