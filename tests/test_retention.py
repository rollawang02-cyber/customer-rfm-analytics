import pandas as pd

from rfm.data import clean_transactions
from rfm.retention import cohort_retention, interpurchase_gaps
from rfm.sql import query_cohort_retention


def test_cohort_retention_matrix(raw_tx):
    ret = cohort_retention(clean_transactions(raw_tx))

    jan, feb = pd.Period("2011-01", "M"), pd.Period("2011-02", "M")
    assert ret.loc[jan, [0, 1, 2]].tolist() == [1.0, 0.0, 0.5]
    assert ret.loc[feb, 0] == 1.0


def test_sql_cohorts_agree_with_pandas(raw_tx):
    ret = cohort_retention(clean_transactions(raw_tx))
    sql = query_cohort_retention(raw_tx)

    for row in sql.itertuples():
        cohort = pd.Period(row.cohort_month, "M")
        assert ret.loc[cohort, row.period] == row.retention


def test_interpurchase_gaps_use_distinct_purchase_days(raw_tx):
    gaps = interpurchase_gaps(clean_transactions(raw_tx))

    assert sorted(gaps.loc[gaps["CustomerID"] == "A", "gap_days"]) == [74]
    assert sorted(gaps.loc[gaps["CustomerID"] == "C", "gap_days"]) == [23]
    assert "B" not in set(gaps["CustomerID"])  # one-time buyer has no gap
