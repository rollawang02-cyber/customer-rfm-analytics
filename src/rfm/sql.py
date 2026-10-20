"""Run the SQL layer (DuckDB) against an in-memory DataFrame.

The SQL files in ``sql/`` are the warehouse-style implementation of the same
feature logic; ``tests/test_sql_parity.py`` asserts they agree with pandas.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from .config import SNAPSHOT, SQL_DIR


def _connect(raw: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("raw_transactions", raw)
    con.execute((SQL_DIR / "01_clean_transactions.sql").read_text())
    return con


def query_rfm(raw: pd.DataFrame, snapshot: pd.Timestamp = SNAPSHOT) -> pd.DataFrame:
    with _connect(raw) as con:
        sql = (SQL_DIR / "02_customer_rfm.sql").read_text()
        out = con.execute(sql, {"snapshot": snapshot.date()}).df()
    return out.set_index("CustomerID")


def query_cohort_retention(raw: pd.DataFrame) -> pd.DataFrame:
    with _connect(raw) as con:
        return con.execute((SQL_DIR / "03_cohort_retention.sql").read_text()).df()
