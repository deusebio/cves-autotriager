"""Shared abstractions for LLM-backed CVE triage comparisons."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from string import Template
from typing import Any

import yaml

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
class TimedResponse:
    """Text response and elapsed wall-clock time for one model invocation."""

    response: str
    execution_time: float


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

        parsed = yaml.safe_load(self._extract_yaml_document(self.comparison))
        if not isinstance(parsed, list):
            raise ValueError("YAML output must be a list of dictionaries")

        rows = [self._yaml_row_to_table_values(row) for row in parsed]
        if rows:
            table.insert(*rows)

    @staticmethod
    def _extract_yaml_document(text: str) -> str:
        stripped = text.strip()
        lines = stripped.splitlines()

        for index, line in enumerate(lines):
            if line.strip().startswith("```"):
                yaml_lines = []
                for fenced_line in lines[index + 1 :]:
                    if fenced_line.strip() == "```":
                        break
                    yaml_lines.append(fenced_line)
                if yaml_lines:
                    return "\n".join(yaml_lines).strip()

        for index, line in enumerate(lines):
            if line.lstrip().startswith("- "):
                return "\n".join(lines[index:]).strip()

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


class BaseModelComparisonChain(WithLogging, ABC):
    """Common cache, prompt-building, and result assembly for comparison chains."""

    def __init__(
        self,
        candidate_model_names: list[str],
        judge_model_name: str,
        *,
        candidate_table: Table | None = None,
        judge_table: Table | None = None,
        comparison_template_name: str = "comparison_prompt.txt",
        output_format: str = "markdown",
    ) -> None:
        if len(candidate_model_names) < 2:
            raise ValueError("At least two candidate models are required")
        if len(candidate_model_names) != len(set(candidate_model_names)):
            raise ValueError("Candidate model names must be unique")

        self._candidate_model_names = candidate_model_names
        self._judge_model_name = judge_model_name
        self._candidate_table = candidate_table
        self._judge_table = judge_table
        self._output_format = output_format
        self._validate_response_table(self._candidate_table)
        self._validate_response_table(self._judge_table)

        template_path = resources.files("cves_autotriager.resources").joinpath(
            comparison_template_name
        )
        self._template = Template(template_path.read_text(encoding="utf-8"))

    def invoke(self, prompt: str, cve_id: str = "unknown") -> ComparisonResult:
        """Synchronously run the comparison workflow."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(prompt, cve_id))
        raise RuntimeError("Use ainvoke() when calling from an active asyncio event loop")

    async def ainvoke(self, prompt: str, cve_id: str = "unknown") -> ComparisonResult:
        """Generate candidate answers, compare them, and return a result object."""
        responses = await self._get_or_invoke_candidates(prompt, cve_id)
        comparison_prompt = self._build_comparison_prompt(prompt, responses)
        comparison = await self._get_or_invoke_judge(comparison_prompt, cve_id)
        return ComparisonResult(prompt, responses, comparison, self._output_format)

    @staticmethod
    def _ensure_table(database: Database | None, table_name: str) -> Table:
        if database is None:
            raise ValueError("A database is required to store model responses")
        if table_name not in database.tables:
            return database.create_table(table_name, RESPONSE_TABLE_SCHEMA)
        return database.get_table(table_name)

    async def _get_or_invoke_candidates(self, prompt: str, cve_id: str) -> dict[str, str]:
        cached_responses: dict[str, str] = {}
        pending_model_names: list[str] = []
        for model_name in self._candidate_model_names:
            cached = self._get_cached_response(
                self._candidate_table,
                model_name=model_name,
                cve_id=cve_id,
                prompt=prompt,
            )
            if cached is None:
                pending_model_names.append(model_name)
            else:
                cached_responses[model_name] = cached

        if pending_model_names:
            pending_results = await self._invoke_candidate_models(pending_model_names, prompt)
            for model_name, result in pending_results.items():
                cached_responses[model_name] = result.response
                self._store_response(
                    self._candidate_table,
                    model_name=model_name,
                    cve_id=cve_id,
                    prompt=prompt,
                    response=result.response,
                    execution_time=result.execution_time,
                )

        return {
            model_name: cached_responses[model_name]
            for model_name in self._candidate_model_names
            if model_name in cached_responses
        }

    def _build_comparison_prompt(self, prompt: str, responses: dict[str, str]) -> str:
        return self._template.substitute(
            original_prompt=prompt,
            responses=json.dumps(responses, indent=2, sort_keys=True),
            table_format=self._output_format,
        )

    async def _get_or_invoke_judge(self, comparison_prompt: str, cve_id: str) -> str:
        cached = self._get_cached_response(
            self._judge_table,
            model_name=self._judge_model_name,
            cve_id=cve_id,
            prompt=comparison_prompt,
        )
        if cached is not None:
            return cached

        result = await self._invoke_judge_model(comparison_prompt)
        self._store_response(
            self._judge_table,
            model_name=self._judge_model_name,
            cve_id=cve_id,
            prompt=comparison_prompt,
            response=result.response,
            execution_time=result.execution_time,
        )
        return result.response

    @abstractmethod
    async def _invoke_candidate_models(
        self, model_names: list[str], prompt: str
    ) -> dict[str, TimedResponse]:
        """Invoke pending candidate models and return responses by model name."""

    @abstractmethod
    async def _invoke_judge_model(self, prompt: str) -> TimedResponse:
        """Invoke the judge model and return its timed response."""

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
    def _validate_response_table(table: Table | None) -> None:
        if table is None:
            return
        if table.schema != RESPONSE_TABLE_SCHEMA:
            raise ValueError(
                f"Response table schema mismatch: expected {RESPONSE_TABLE_SCHEMA}, "
                f"found {table.schema}"
            )
