import pandas as pd
import pytest


def _row(inv, code, qty, date, price, cust, country="United Kingdom", desc="ITEM"):
    return {
        "InvoiceNo": inv,
        "StockCode": code,
        "Description": desc,
        "Quantity": qty,
        "InvoiceDate": pd.Timestamp(date),
        "UnitPrice": price,
        "CustomerID": cust,
        "Country": country,
    }


@pytest.fixture
def raw_tx():
    """A hand-built slice of the Online Retail schema.

    Customer A: two orders (Jan, Mar) and a partial return of the first one.
    Customer B: one order in Jan that also carries a postage line.
    Customer C: two orders in Feb.
    Noise: an exact duplicate line, a guest (no CustomerID) line and a free
    (UnitPrice 0) line - all of which must not reach the customer table.
    """
    rows = [
        _row("1001", "85123A", 10, "2011-01-05 10:00", 2.0, "A"),
        _row("1001", "85123A", 10, "2011-01-05 10:00", 2.0, "A"),  # exact duplicate
        _row("1001", "71053", 5, "2011-01-05 10:00", 4.0, "A"),
        _row("C1002", "71053", -2, "2011-01-10 09:00", 4.0, "A"),  # return
        _row("1003", "85123A", 3, "2011-03-20 15:00", 2.0, "A"),
        _row("1004", "22423", 1, "2011-01-15 11:00", 12.0, "B", "France"),
        _row("1004", "POST", 1, "2011-01-15 11:00", 18.0, "B", "France"),  # postage
        _row("1005", "84029G", 4, "2011-02-02 08:00", 5.0, "C"),
        _row("1006", "84029G", 2, "2011-02-25 08:00", 5.0, "C"),
        _row("1007", "85123A", 50, "2011-02-10 12:00", 2.0, None),  # guest checkout
        _row("1008", "22423", 1, "2011-02-11 12:00", 0.0, "C"),  # free item
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def snapshot():
    return pd.Timestamp("2011-04-01")
