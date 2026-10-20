from rfm.data import clean_transactions


def test_cleaning_keeps_only_identified_merchandise_lines(raw_tx):
    tx = clean_transactions(raw_tx)

    assert len(tx) == 7  # 11 raw - duplicate - postage - guest - free item
    assert set(tx["CustomerID"]) == {"A", "B", "C"}
    assert "POST" not in set(tx["StockCode"])


def test_cleaning_keeps_returns_as_negative_revenue(raw_tx):
    tx = clean_transactions(raw_tx)

    returns = tx[tx["IsCancel"]]
    assert list(returns["InvoiceNo"]) == ["C1002"]
    assert returns["Revenue"].sum() == -8.0
