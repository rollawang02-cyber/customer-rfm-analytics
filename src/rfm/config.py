"""Project-wide constants: paths, business rules and analysis dates.

Every analytical choice that changes a number in the report lives here, so a
reviewer can audit (or change) the assumptions in one place.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "online_retail.zip"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "reports" / "figures"
SQL_DIR = ROOT / "sql"

# Stock codes that are not merchandise: postage, fees, manual adjustments,
# discounts, samples, bad-debt write-offs and gift vouchers. Counting them as
# "customer spend" inflates Monetary (postage) or deflates it (Amazon fees).
NON_MERCH_CODES = frozenset(
    {"POST", "DOT", "M", "m", "C2", "D", "S", "BANK CHARGES", "AMAZONFEE", "CRUK", "B", "PADS"}
)
NON_MERCH_PREFIXES = ("gift_",)

# The data runs 2010-12-01 08:26 -> 2011-12-09 12:50. The RFM snapshot is the
# day after the last invoice (the tutorial used an arbitrary 2012-01-01, which
# adds ~3 weeks (21 days for most customers) of phantom recency).
SNAPSHOT = pd.Timestamp("2011-12-10")

# Calibration / hold-out split for validating the CLV model: the last 13 weeks
# are held out and never seen during fitting.
CAL_END = pd.Timestamp("2011-09-09 23:59:59")
HOLDOUT_DAYS = 91

# Forecast horizon for customer lifetime value (days).
CLV_HORIZON_DAYS = 180

RANDOM_STATE = 42
