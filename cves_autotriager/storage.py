"""SQLite database and table abstractions."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Generator, Sequence
from contextlib import AbstractContextManager, closing, contextmanager
from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd

DataType: TypeAlias = int | float | str | bytes | None
SchemaType: TypeAlias = type[int] | type[float] | type[str] | type[bytes]

TYPES: dict[SchemaType, str] = {
    int: "INTEGER",
    float: "REAL",
    str: "TEXT",
    bytes: "BLOB",
}
ITYPES: dict[str, SchemaType] = {value: key for key, value in TYPES.items()}


class TableExists(Exception):
    """Raised when a table already exists."""


class TableNotFound(Exception):
    """Raised when a table does not exist."""


class DatabaseNotFound(Exception):
    """Raised when a database does not exist."""


def _identifier(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"Invalid SQLite identifier: {value!r}")
    return f'"{value}"'


class SQLiteClient:
    """Manage local SQLite database files."""

    def __init__(self, root_path: str | Path = ".") -> None:
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

    @property
    def databases(self) -> list[str]:
        """List database names in the configured directory."""
        return sorted(path.stem for path in self.root_path.glob("*.db") if path.is_file())

    def get_database(self, name: str) -> Database:
        """Get a database, creating its file if necessary."""
        database = Database(name, self)
        with database.connection:
            pass
        return database

    def open_database(self, name: str) -> Database:
        """Get an existing database by name."""
        if name not in self.databases:
            raise DatabaseNotFound(name)
        return Database(name, self)


class Database:
    """A SQLite database file."""

    def __init__(self, name: str, client: SQLiteClient) -> None:
        _identifier(name)
        self.client = client
        self.name = name

    @property
    def path(self) -> Path:
        """Return the database file path."""
        return self.client.root_path / f"{self.name}.db"

    @property
    def connection(self) -> AbstractContextManager[sqlite3.Connection]:
        """Open a transactional connection to this database."""
        return self._connection()

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        with closing(sqlite3.connect(self.path)) as connection, connection:
            yield connection

    @property
    def tables(self) -> list[str]:
        """List user-created tables in this database."""
        with self.connection as connection:
            rows = connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def get_table(self, name: str) -> Table:
        """Get an existing table and its schema."""
        if name not in self.tables:
            raise TableNotFound(name)
        with self.connection as connection:
            rows = connection.execute(f"PRAGMA table_info({_identifier(name)})").fetchall()
        schema = [(str(row[1]), ITYPES[str(row[2]).upper()]) for row in rows]
        return Table(name, self, schema)

    def create_table(self, name: str, schema: list[tuple[str, SchemaType]]) -> Table:
        """Create a table from a name and typed schema."""
        if name in self.tables:
            raise TableExists(name)
        if not schema:
            raise ValueError("Table schema cannot be empty")
        columns = ", ".join(
            f"{_identifier(column)} {TYPES[column_type]}" for column, column_type in schema
        )
        with self.connection as connection:
            connection.execute(f"CREATE TABLE {_identifier(name)} ({columns})")
        return Table(name, self, schema)

    def drop(self) -> None:
        """Delete this database file."""
        if not self.path.exists():
            raise DatabaseNotFound(self.name)
        self.path.unlink()


class Table:
    """A typed SQLite table."""

    def __init__(
        self,
        name: str,
        database: Database,
        schema: list[tuple[str, SchemaType]],
    ) -> None:
        _identifier(name)
        self.name = name
        self.database = database
        self.schema = schema

    def validate(self, values: Sequence[DataType]) -> dict[str, DataType]:
        """Validate and map a row against the table schema."""
        if len(values) != len(self.schema):
            raise ValueError(f"Expected {len(self.schema)} values, found {len(values)}")
        output: dict[str, DataType] = {}
        for value, (column, expected_type) in zip(values, self.schema, strict=True):
            if value is not None and not isinstance(value, expected_type):
                raise TypeError(
                    f"Column {column}: expected {expected_type.__name__}, "
                    f"found {type(value).__name__}"
                )
            output[column] = value
        return output

    def drop(self) -> None:
        """Drop this table."""
        with self.database.connection as connection:
            connection.execute(f"DROP TABLE {_identifier(self.name)}")

    def rows(self) -> Generator[dict[str, DataType], None, None]:
        """Iterate over rows as mappings keyed by column name."""
        with self.database.connection as connection:
            cursor = connection.execute(f"SELECT * FROM {_identifier(self.name)}")
            for row in cursor:
                yield self.validate(row)

    def find_one(self, column: str, value: DataType) -> dict[str, DataType] | None:
        """Return the first row matching a column value."""
        columns = {name for name, _ in self.schema}
        if column not in columns:
            raise ValueError(f"Column not found in table {self.name}: {column}")
        statement = (
            f"SELECT * FROM {_identifier(self.name)} "
            f"WHERE {_identifier(column)} = ? LIMIT 1"
        )
        with self.database.connection as connection:
            row = connection.execute(statement, (value,)).fetchone()
        return None if row is None else self.validate(row)

    def insert(self, *rows: Sequence[DataType]) -> int:
        """Insert validated rows using one statement per row."""
        placeholders = ", ".join("?" for _ in self.schema)
        statement = f"INSERT INTO {_identifier(self.name)} VALUES ({placeholders})"
        with self.database.connection as connection:
            for row in rows:
                self.validate(row)
                connection.execute(statement, tuple(row))
        return len(rows)


def write_dataframe_to_sqlite(
    dataframe: pd.DataFrame,
    database_path: str | Path,
    *,
    table: str = "cves",
    if_exists: Literal["append", "replace", "fail"] = "replace",
) -> int:
    """Write an ID-column dataframe to SQLite one row per insert."""
    if "id" not in dataframe.columns:
        raise ValueError("Dataframe must contain an 'id' column")

    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database) as connection:
        dataframe.to_sql(
            table,
            connection,
            if_exists=if_exists,
            index=False,
            chunksize=1,
        )
    return len(dataframe)