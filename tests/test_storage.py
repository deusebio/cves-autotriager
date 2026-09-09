import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from cves_autotriager.storage import (
    DatabaseNotFound,
    SQLiteClient,
    TableExists,
    TableNotFound,
    write_dataframe_to_sqlite,
)


def test_database_and_table_lifecycle(tmp_path: Path) -> None:
    client = SQLiteClient(tmp_path)
    database = client.get_database("triage")
    table = database.create_table(
        "cves",
        [("id", str), ("severity", str), ("score", float)],
    )

    rows_inserted = table.insert(
        ["CVE-2026-1234", "HIGH", 8.1],
        ["CVE-2026-5678", "LOW", None],
    )

    assert client.databases == ["triage"]
    assert database.tables == ["cves"]
    assert rows_inserted == 2
    assert list(database.get_table("cves").rows()) == [
        {"id": "CVE-2026-1234", "severity": "HIGH", "score": 8.1},
        {"id": "CVE-2026-5678", "severity": "LOW", "score": None},
    ]

    table.drop()
    assert database.tables == []
    database.drop()
    assert client.databases == []


def test_database_and_table_errors(tmp_path: Path) -> None:
    client = SQLiteClient(tmp_path)
    database = client.get_database("triage")
    database.create_table("cves", [("id", str)])

    with pytest.raises(TableExists):
        database.create_table("cves", [("id", str)])
    with pytest.raises(TableNotFound):
        database.get_table("missing")
    with pytest.raises(DatabaseNotFound):
        client.open_database("missing")


def test_table_validates_row_shape_and_types(tmp_path: Path) -> None:
    table = SQLiteClient(tmp_path).get_database("triage").create_table(
        "cves", [("id", str), ("score", float)]
    )

    with pytest.raises(ValueError, match="Expected 2 values"):
        table.insert(["CVE-2026-1234"])
    with pytest.raises(TypeError, match="Column score"):
        table.insert(["CVE-2026-1234", "high"])


def test_table_finds_one_row_by_column(tmp_path: Path) -> None:
    table = SQLiteClient(tmp_path).get_database("triage").create_table(
        "cves", [("id", str), ("severity", str)]
    )
    table.insert(["CVE-2026-1234", "HIGH"])

    assert table.find_one("id", "CVE-2026-1234") == {
        "id": "CVE-2026-1234",
        "severity": "HIGH",
    }
    assert table.find_one("id", "CVE-2026-9999") is None


def test_write_dataframe_preserves_id_column_and_rows(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        [
            {"id": "CVE-2026-1234", "severity": "HIGH", "image": "image-one"},
            {"id": "CVE-2026-1234", "severity": "HIGH", "image": "image-two"},
        ]
    )
    database_path = tmp_path / "data" / "cves.db"

    rows_written = write_dataframe_to_sqlite(dataframe, database_path)

    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT id, severity, image FROM cves ORDER BY image"
        ).fetchall()
    assert rows_written == 2
    assert rows == [
        ("CVE-2026-1234", "HIGH", "image-one"),
        ("CVE-2026-1234", "HIGH", "image-two"),
    ]


def test_write_dataframe_can_append(tmp_path: Path) -> None:
    database_path = tmp_path / "cves.db"
    first = pd.DataFrame([{"id": "CVE-1", "severity": "HIGH"}])
    second = pd.DataFrame([{"id": "CVE-2", "severity": "LOW"}])

    write_dataframe_to_sqlite(first, database_path)
    write_dataframe_to_sqlite(second, database_path, if_exists="append")

    with sqlite3.connect(database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM cves").fetchone()
    assert count == (2,)


def test_write_dataframe_requires_id_column(tmp_path: Path) -> None:
    dataframe = pd.DataFrame([{"severity": "HIGH"}])

    with pytest.raises(ValueError, match="contain an 'id' column"):
        write_dataframe_to_sqlite(dataframe, tmp_path / "cves.db")