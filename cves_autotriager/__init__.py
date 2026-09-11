"""cves-autotriager: automated triage tooling for CVEs."""

from cves_autotriager.llm.base import ComparisonResult
from cves_autotriager.llm.langchain import ModelComparisonChain
from cves_autotriager.parser import NVDEnricher, TrivyReportParser
from cves_autotriager.prompt import CVEPromptBuilder
from cves_autotriager.storage import (
    Database,
    DatabaseNotFound,
    SQLiteClient,
    Table,
    TableExists,
    TableNotFound,
    write_dataframe_to_sqlite,
)

__version__ = "0.1.0"

__all__ = [
    "CVEPromptBuilder",
    "ComparisonResult",
    "Database",
    "DatabaseNotFound",
    "ModelComparisonChain",
    "NVDEnricher",
    "SQLiteClient",
    "Table",
    "TableExists",
    "TableNotFound",
    "TrivyReportParser",
    "write_dataframe_to_sqlite",
    "__version__",
]
