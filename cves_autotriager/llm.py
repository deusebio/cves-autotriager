"""Run and compare CVE triage responses from LangChain chat models."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from string import Template
from typing import Any

import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel

from cves_autotriager.logging_utils import WithLogging
from cves_autotriager.storage import Database, SchemaType, Table

RESPONSE_TABLE_SCHEMA: list[tuple[str, SchemaType]] = [
    ("timestamp", str),
    ("model_name", str),
    ("cve_id", str),
    ("prompt", str),
    ("response", str),
    ("execution_time", float),
]

OUTPUT_COLUMNS = [
    "image",
    "classification",
    "rationale",
    "controls",
    "confidence",
    "best_model",
]

OUTPUT_TABLE_SCHEMA: list[tuple[str, SchemaType]] = [(column, str) for column in OUTPUT_COLUMNS]


@dataclass(frozen=True)
class ComparisonResult:
    """Candidate model responses and the judge model's comparison."""

    prompt: str
    responses: dict[str, str]
    comparison: str
    output: str

    def write(self, table: Table) -> None:
        if self.output != "yaml":
            raise ValueError("ComparisonResult.write only supports yaml output")

        table_columns = [column for column, _ in table.schema]
        if table_columns != OUTPUT_COLUMNS:
            raise ValueError(
                f"Output table schema mismatch: expected {OUTPUT_COLUMNS}, "
                f"found {table_columns}"
            )

        parsed = yaml.safe_load(self._strip_code_fence(self.comparison))
        if not isinstance(parsed, list):
            raise ValueError("YAML output must be a list of dictionaries")

        rows = [self._yaml_row_to_table_values(row) for row in parsed]
        if rows:
            table.insert(*rows)

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
        return stripped

    @staticmethod
    def _yaml_row_to_table_values(row: object) -> list[str | None]:
        if not isinstance(row, dict):
            raise ValueError("YAML output rows must be dictionaries")

        missing_columns = sorted(set(OUTPUT_COLUMNS) - set(row))
        if missing_columns:
            raise ValueError(f"YAML output is missing columns: {', '.join(missing_columns)}")

        return [ComparisonResult._yaml_value(row[column]) for column in OUTPUT_COLUMNS]

    @staticmethod
    def _yaml_value(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)


class ModelComparisonChain(WithLogging):
    """Send one prompt to candidate models and compare their responses."""

    def __init__(
        self,
        models: dict[str, BaseChatModel],
        judge: BaseChatModel,
        *,
        database: Database | None = None,
        candidate_table_name: str = "candidate_responses",
        judge_table_name: str = "judge_responses",
        comparison_template_name: str = "comparison_prompt.txt",
        output_format: str = "markdown",
    ) -> None:
        if len(models) < 2:
            raise ValueError("At least two candidate models are required")
        if len(models) != len(set(models)):
            raise ValueError("Candidate model names must be unique")

        self._output_format = output_format

        template_path = resources.files("cves_autotriager.resources").joinpath(
            comparison_template_name
        )
        self._template = Template(template_path.read_text(encoding="utf-8"))
        self._candidate_models = dict(models)
        self._candidates = RunnableParallel(self._candidate_models)
        self._judge = judge

        # Initialize database and tables
        self._database = database
        self._candidate_table = (
            self._ensure_table(database, candidate_table_name) if database is not None else None
        )
        self._judge_table = (
            self._ensure_table(database, judge_table_name) if database is not None else None
        )

    def invoke(self, prompt: str, cve_id: str = "unknown") -> ComparisonResult:
        """Generate candidate answers in parallel, then invoke the judge."""
        cached_responses: dict[str, str] = {}
        pending_model_names = []
        for name in self._candidate_models:
            cached = self._get_cached_response(
                self._candidate_table,
                model_name=name,
                cve_id=cve_id,
                prompt=prompt,
            )
            if cached is not None:
                cached_responses[name] = cached
            else:
                pending_model_names.append(name)

        if pending_model_names:
            timed_models = {
                name: RunnableLambda(self._timed_invoke(model))
                for name, model in self._candidate_models.items()
                if name in pending_model_names
            }
            raw_results = RunnableParallel(timed_models).invoke(prompt)
            for name, result in raw_results.items():
                if "response" in result:
                    response = result["response"]
                    execution_time = float(result["execution_time"])
                    self._store_response(
                        self._candidate_table,
                        model_name=name,
                        cve_id=cve_id,
                        prompt=prompt,
                        response=response,
                        execution_time=execution_time,
                    )
                    cached_responses[name] = response
                elif "error" in result:
                    self.logger.error(f"Error invoking model {name}: {result['error']}")

        responses = cached_responses
        comparison_prompt = self._template.substitute(
            original_prompt=prompt,
            responses=json.dumps(responses, indent=2, sort_keys=True),
            table_format=self._output_format,
        )

        comparison = self._get_or_invoke_judge(comparison_prompt, cve_id)
        return ComparisonResult(prompt, responses, comparison, self._output_format)

    def _timed_invoke(self, model: BaseChatModel) -> Callable[[str], dict[str, str | float]]:
        def _run(_prompt: str) -> dict[str, str | float]:
            started = time.perf_counter()
            try:
                raw_response = model.invoke(_prompt)
            except Exception as error:
                return {"error": str(error)}

            execution_time = time.perf_counter() - started
            response = self._message_text(raw_response)
            return {
                "response": response,
                "execution_time": execution_time,
            }

        return _run

    def _get_or_invoke_judge(
        self,
        comparison_prompt: str,
        cve_id: str,
    ) -> str:
        cached = self._get_cached_response(
            self._judge_table,
            model_name=self._judge_name,
            cve_id=cve_id,
            prompt=comparison_prompt,
        )
        if cached is not None:
            return cached

        started = time.perf_counter()
        comparison = self._message_text(self._judge.invoke(comparison_prompt))
        execution_time = time.perf_counter() - started
        self._store_response(
            self._judge_table,
            model_name=self._judge_name,
            cve_id=cve_id,
            prompt=comparison_prompt,
            response=comparison,
            execution_time=execution_time,
        )
        return comparison

    @property
    def _judge_name(self) -> str:
        for attr in ("model_name", "model", "name"):
            value = getattr(self._judge, attr, None)
            if isinstance(value, str) and value:
                return value
        return "judge"

    @staticmethod
    def _ensure_table(database: Database | None, table_name: str) -> Table:
        if database is None:
            raise ValueError("A database is required to store model responses")
        if table_name not in database.tables:
            return database.create_table(table_name, RESPONSE_TABLE_SCHEMA)
        return database.get_table(table_name)

    def _get_cached_response(
        self,
        table: Table | None,
        *,
        model_name: str,
        cve_id: str,
        prompt: str,
    ) -> str | None:
        if table is None:
            return None
        self.logger.debug("Fetching cached response for model %s and CVE %s", model_name, cve_id)
        with table.database.connection as connection:
            row = connection.execute(
                f'SELECT response FROM "{table.name}" '
                "WHERE model_name = ? AND cve_id = ? AND prompt = ? "
                "ORDER BY rowid DESC LIMIT 1",
                (model_name, cve_id, prompt),
            ).fetchone()
        return None if row is None else str(row[0])

    def _store_response(
        self,
        table: Table | None,
        *,
        model_name: str,
        cve_id: str,
        prompt: str,
        response: str,
        execution_time: float,
    ) -> None:
        if table is None:
            return

        self.logger.info("Storing response for model %s and CVE %s", model_name, cve_id)
        table.insert(
            (
                datetime.now(timezone.utc).isoformat(),
                model_name,
                cve_id,
                prompt,
                response,
                float(execution_time),
            )
        )

    @staticmethod
    def _message_text(message: object) -> str:
        if not isinstance(message, BaseMessage):
            raise TypeError(f"Expected a LangChain message, found {type(message).__name__}")
        return message.text
