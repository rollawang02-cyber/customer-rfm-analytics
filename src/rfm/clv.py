"""Probabilistic customer lifetime value: BG/NBD + Gamma-Gamma.

Implemented from the closed-form likelihoods in
Fader, Hardie & Lee (2005), "Counting Your Customers the Easy Way", and
Fader, Hardie & Lee (2005), "RFM and CLV: Using Iso-Value Curves".
Time unit throughout is days; a "transaction" is a distinct purchase day.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, hyp2f1, logsumexp


def _purchase_days(tx: pd.DataFrame, obs_end: pd.Timestamp) -> pd.DataFrame:
    """Customer-days with a purchase, valued net of same-day cancellations.

    A day whose purchases were cancelled the same day (e.g. customer 16446's
    80,995-unit order, cancelled 12 minutes later) is not a transaction; left
    in, it would hand that customer a six-figure CLV.
    """
    lines = tx[tx["InvoiceDate"] < obs_end].assign(day=lambda d: d["InvoiceDate"].dt.normalize())
    days = lines.groupby(["CustomerID", "day"]).agg(
        Revenue=("Revenue", "sum"), has_purchase=("IsCancel", lambda c: (~c).any())
    )
    days = days[days["has_purchase"] & (days["Revenue"] > 0)]
    return days.drop(columns="has_purchase").reset_index().sort_values(["CustomerID", "day"])


def customer_summary(tx: pd.DataFrame, obs_end: pd.Timestamp) -> pd.DataFrame:
    """BG/NBD inputs per customer for purchases strictly before ``obs_end``.

    ``x`` repeat purchase days, ``t_x`` day of last purchase measured from the
    first, ``T`` days from first purchase to ``obs_end``, ``monetary_value``
    mean value of the *repeat* purchase days (Gamma-Gamma convention).
    """
    days = _purchase_days(tx, obs_end)
    g = days.groupby("CustomerID")
    first, last = g["day"].min(), g["day"].max()
    out = pd.DataFrame(
        {
            "x": g.size() - 1,
            "t_x": (last - first).dt.days,
            "T": (obs_end.normalize() - first).dt.days,
        }
    )
    days["is_first"] = days["day"].eq(days["CustomerID"].map(first))
    repeat_mean = days[~days["is_first"]].groupby("CustomerID")["Revenue"].mean()
    out["monetary_value"] = repeat_mean.reindex(out.index).fillna(0.0)
    return out


def holdout_purchases(tx: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Distinct purchase days per customer in ``[start, end)``."""
    days = _purchase_days(tx, end)
    return days[days["day"] >= start.normalize()].groupby("CustomerID").size()


@dataclass
class BetaGeoModel:
    """BG/NBD: Poisson purchasing with Gamma-heterogeneous rates and a
    Beta-distributed probability of dropping out after each purchase."""

    params_: dict = field(default_factory=dict)

    @staticmethod
    def _loglik(log_params, x, t_x, T):
        r, alpha, a, b = np.exp(log_params)
        a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        a3 = -(r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            a4 = np.where(x > 0, np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + t_x), -np.inf)
        return a1 + a2 + logsumexp(np.vstack([a3, a4]), axis=0)

    def fit(self, x, t_x, T):
        x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
        x0 = np.log([1.0, max(T.mean() / max(x.mean(), 1e-3), 1.0), 1.0, 1.0])
        res = minimize(lambda p: -self._loglik(p, x, t_x, T).sum(), x0, method="L-BFGS-B",
                       bounds=[(-10, 10)] * 4)
        if not res.success:
            raise RuntimeError(f"BG/NBD did not converge: {res.message}")
        self.params_ = dict(zip(["r", "alpha", "a", "b"], np.exp(res.x)))
        self.loglik_ = -res.fun
        return self

    def p_alive(self, x, t_x, T):
        r, alpha, a, b = (self.params_[k] for k in ("r", "alpha", "a", "b"))
        x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
        with np.errstate(divide="ignore", invalid="ignore"):
            odds = np.where(x > 0, a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
        return 1.0 / (1.0 + odds)

    def expected_purchases(self, t, x, t_x, T):
        """E[# purchase days in (T, T + t] | x, t_x, T]."""
        r, alpha, a, b = (self.params_[k] for k in ("r", "alpha", "a", "b"))
        x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        num = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
        with np.errstate(divide="ignore", invalid="ignore"):
            den = 1 + np.where(x > 0, a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
        return num / den


@dataclass
class GammaGammaModel:
    """Gamma-Gamma spend model for customers with at least one repeat purchase."""

    params_: dict = field(default_factory=dict)

    @staticmethod
    def _loglik(log_params, x, m):
        p, q, g = np.exp(log_params)
        return (
            gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(g)
            + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(g + m * x)
        )

    def fit(self, x, m):
        x, m = np.asarray(x, dtype=float), np.asarray(m, dtype=float)
        res = minimize(lambda p: -self._loglik(p, x, m).sum(), np.log([1.0, 2.0, m.mean()]),
                       method="L-BFGS-B", bounds=[(-10, 15)] * 3)
        if not res.success:
            raise RuntimeError(f"Gamma-Gamma did not converge: {res.message}")
        self.params_ = dict(zip(["p", "q", "gamma"], np.exp(res.x)))
        return self

    def expected_spend(self, x, m):
        """E[value of a future purchase day | x repeat days with mean m]."""
        p, q, g = (self.params_[k] for k in ("p", "q", "gamma"))
        x, m = np.asarray(x, dtype=float), np.asarray(m, dtype=float)
        population = p * g / (q - 1)
        return np.where(x > 0, (p * g + p * x * m) / (p * x + q - 1), population)
