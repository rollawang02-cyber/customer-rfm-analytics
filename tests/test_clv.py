import numpy as np
import pandas as pd
import pytest

from rfm.clv import BetaGeoModel, GammaGammaModel, customer_summary
from rfm.data import clean_transactions


def simulate_bgnbd(n, r, alpha, a, b, seed=0):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(r, 1 / alpha, n)
    p = rng.beta(a, b, n)
    T = rng.uniform(150, 365, n)
    x, t_x = np.zeros(n), np.zeros(n)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / lam[i])
            if t > T[i]:
                break
            x[i] += 1
            t_x[i] = t
            if rng.random() < p[i]:
                break
    return x, t_x, T


def test_bgnbd_recovers_known_parameters():
    true = dict(r=0.5, alpha=20.0, a=0.8, b=2.5)
    x, t_x, T = simulate_bgnbd(8000, **true)

    m = BetaGeoModel().fit(x, t_x, T)

    assert m.params_["r"] == pytest.approx(true["r"], rel=0.15)
    assert m.params_["alpha"] == pytest.approx(true["alpha"], rel=0.15)
    mean_dropout = m.params_["a"] / (m.params_["a"] + m.params_["b"])
    assert mean_dropout == pytest.approx(0.8 / 3.3, rel=0.2)


def test_gamma_gamma_recovers_population_mean_spend():
    rng = np.random.default_rng(1)
    p, q, gamma = 6.0, 4.0, 15.0
    nu = rng.gamma(q, 1 / gamma, 5000)
    x = rng.integers(1, 10, 5000)
    m_x = np.array([rng.gamma(p, 1 / v, k).mean() for v, k in zip(nu, x)])

    gg = GammaGammaModel().fit(x, m_x)

    pop_mean = gg.params_["p"] * gg.params_["gamma"] / (gg.params_["q"] - 1)
    assert pop_mean == pytest.approx(p * gamma / (q - 1), rel=0.1)


def test_customer_summary_uses_repeat_purchase_days(raw_tx, snapshot):
    s = customer_summary(clean_transactions(raw_tx), obs_end=snapshot)

    assert s.loc["A", ["x", "t_x", "T"]].tolist() == [1, 74, 86]
    assert s.loc["A", "monetary_value"] == 6.0  # mean value of the repeat purchase day
    assert s.loc["B", ["x", "t_x"]].tolist() == [0, 0]
