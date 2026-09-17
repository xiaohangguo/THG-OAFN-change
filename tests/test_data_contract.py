import pandas as pd
import pytest

from finrisk.data_contract import SchemaError, laundering_mask, resolve_transaction_columns


def test_resolves_ibm_transaction_schema():
    columns = ["Timestamp", "From Bank", "Account", "To Bank", "Account.1", "Is Laundering"]
    result = resolve_transaction_columns(columns)
    assert result["timestamp"] == "Timestamp"
    assert result["destination_account"] == "Account.1"


def test_missing_timestamp_stops_experiment():
    with pytest.raises(SchemaError, match="timestamp"):
        resolve_transaction_columns(["From Bank", "Account", "To Bank", "Account.1", "Is Laundering"])


def test_label_normalization_accepts_binary_and_boolean_values():
    values = pd.Series([0, "1", "false", "yes"])
    assert laundering_mask(values).tolist() == [False, True, False, True]


def test_unknown_labels_stop_experiment():
    with pytest.raises(SchemaError, match="Unsupported"):
        laundering_mask(pd.Series(["legitimate", "laundering"]))
