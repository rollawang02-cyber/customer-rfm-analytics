import pandas as pd

from rfm.data import clean_transactions
from rfm.features import build_rfm
from rfm.sql import query_rfm

COLS = ["recency_days", "frequency", "monetary", "gross_revenue", "tenure_days"]


def test_sql_pipeline_reproduces_pandas_rfm(raw_tx, snapshot):
    expected = build_rfm(clean_transactions(raw_tx), snapshot)[COLS]
    got = query_rfm(raw_tx, snapshot)[COLS]

    pd.testing.assert_frame_equal(got.sort_index(), expected.sort_index(), check_dtype=False)
