import pytest

from rfm.data import clean_transactions
from rfm.features import build_rfm


def test_rfm_table_matches_hand_worked_example(raw_tx, snapshot):
    rfm = build_rfm(clean_transactions(raw_tx), snapshot)

    a = rfm.loc["A"]
    assert a["recency_days"] == 12
    assert a["frequency"] == 2
    assert a["gross_revenue"] == 46.0
    assert a["monetary"] == 38.0  # net of the 8.0 return
    assert a["tenure_days"] == 86
    assert a["return_rate"] == pytest.approx(8 / 46)

    assert rfm.loc["B", ["recency_days", "frequency", "monetary"]].tolist() == [76, 1, 12.0]
    assert rfm.loc["B", "country"] == "France"
    assert rfm.loc["C", ["recency_days", "frequency", "monetary"]].tolist() == [35, 2, 30.0]
