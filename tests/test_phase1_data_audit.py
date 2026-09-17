import importlib.util
from pathlib import Path


def load_audit_module():
    script = Path(__file__).parents[1] / "scripts" / "phase1_data_audit.py"
    spec = importlib.util.spec_from_file_location("phase1_data_audit", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_chunked_audit_reports_temporal_financial_fields(tmp_path):
    csv = tmp_path / "transactions.csv"
    csv.write_text(
        "Timestamp,From Bank,Account,To Bank,Account.1,Payment Format,Is Laundering\n"
        "2022-09-01 00:00:00,1,A,2,B,Wire,0\n"
        "2022-09-02 12:00:00,1,A,3,C,Card,1\n",
        encoding="utf-8",
    )
    audit = load_audit_module().audit_transactions(csv, chunksize=1)
    assert audit["rows"] == 2
    assert audit["positive_rows"] == 1
    assert audit["unique_accounts"] == 3
    assert audit["timestamp_min"].startswith("2022-09-01")
