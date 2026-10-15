import asyncio
from importlib import resources
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage

from cves_autotriager.llm import (
    OUTPUT_TABLE_SCHEMA,
    ComparisonResult,
    ModelComparisonChain,
    asafe_save,
    safe_save,
)
from cves_autotriager.storage import SQLiteClient


def test_comparison_chain_fans_out_prompt_and_invokes_judge() -> None:
    first = FakeListChatModel(responses=["first analysis"])
    second = FakeListChatModel(responses=["second analysis"])
    judge = FakeListChatModel(responses=["second is better supported"])

    result = ModelComparisonChain(
        {"first": first, "second": second},
        judge,
    ).invoke("Assess CVE-2026-1234", cve_id="CVE-2026-1234")

    assert result.prompt == "Assess CVE-2026-1234"
    assert result.cve_id == "CVE-2026-1234"
    assert result.responses == {
        "first": "first analysis",
        "second": "second analysis",
    }
    assert result.comparison == "second is better supported"


def test_comparison_prompt_describes_yaml_output_contract() -> None:
    template = (
        resources.files("cves_autotriager.resources")
        .joinpath("comparison_prompt.txt")
        .read_text(encoding="utf-8")
    )

    assert "${table_format}" in template
    assert "If the output format is `YAML`" in template
    assert "classification:" in template
    assert "controls:" in template
    assert "best_model:" in template


def test_comparison_chain_caches_candidate_and_judge_responses(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    first = FakeListChatModel(responses=["first analysis"])
    second = FakeListChatModel(responses=["second analysis"])
    judge = FakeListChatModel(responses=["second is better supported"])

    chain = ModelComparisonChain(
        {"first": first, "second": second},
        judge,
        database=database,
        candidate_table_name="candidate_responses",
        judge_table_name="judge_responses",
    )

    first_result = chain.invoke("Assess CVE-2026-1234", cve_id="CVE-2026-1234")
    second_result = chain.invoke("Assess CVE-2026-1234", cve_id="CVE-2026-1234")

    assert first_result.responses == {
        "first": "first analysis",
        "second": "second analysis",
    }
    assert second_result.responses == first_result.responses
    assert second_result.comparison == first_result.comparison

    candidate_rows = list(database.get_table("candidate_responses").rows())
    judge_rows = list(database.get_table("judge_responses").rows())
    assert len(candidate_rows) == 2
    assert len(judge_rows) == 1
    assert {row["model_name"] for row in candidate_rows} == {"first", "second"}
    assert all(row["cve_id"] == "CVE-2026-1234" for row in candidate_rows + judge_rows)


def test_comparison_chain_ainvoke_runs_and_caches_models(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    first = FakeListChatModel(responses=["first analysis"])
    second = FakeListChatModel(responses=["second analysis"])
    judge = FakeListChatModel(responses=["second is better supported"])
    chain = ModelComparisonChain(
        {"first": first, "second": second},
        judge,
        database=database,
        candidate_table_name="candidate_responses",
        judge_table_name="judge_responses",
    )

    first_result = asyncio.run(chain.ainvoke("Assess CVE-2026-1234", cve_id="CVE-2026-1234"))
    cached_result = asyncio.run(chain.ainvoke("Assess CVE-2026-1234", cve_id="CVE-2026-1234"))

    assert first_result.responses == {
        "first": "first analysis",
        "second": "second analysis",
    }
    assert cached_result.responses == first_result.responses
    assert cached_result.comparison == first_result.comparison
    assert len(list(database.get_table("candidate_responses").rows())) == 2
    assert len(list(database.get_table("judge_responses").rows())) == 1


def test_comparison_result_writes_yaml_output_to_table(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    output = database.create_table("output", OUTPUT_TABLE_SCHEMA)
    result = ComparisonResult(
        cve_id="CVE-2026-1234",
        prompt="Assess CVE-2026-1234",
        responses={},
        comparison_prompt="Compare analyses for CVE-2026-1234",
        comparison=(
            "- image: ubuntu:1.11\n"
            "  classification: Mitigated\n"
            "  rationale: Patched package\n"
            "  controls: Network policy\n"
            "  confidence: High\n"
            "  best_model: first\n"
        ),
        output="yaml",
    )

    result.write(output)

    rows = list(output.rows())
    assert rows == [
        {
            "cve_id": "CVE-2026-1234",
            "image": "ubuntu:1.11",
            "classification": "Mitigated",
            "rationale": "Patched package",
            "controls": "Network policy",
            "confidence": "High",
            "best_model": "first",
        }
    ]


def test_comparison_result_writes_fenced_yaml_with_leading_prefix(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    output = database.create_table("output", OUTPUT_TABLE_SCHEMA)
    result = ComparisonResult(
        cve_id="CVE-2026-4035",
        prompt="Assess CVE-2026-4035",
        responses={},
        comparison_prompt="Compare analyses for CVE-2026-4035",
        comparison=(
            ":\n"
            "```yaml\n"
            "- image: image-one;image-two\n"
            "  classification: False positive\n"
            "  rationale: Package is present but service is not exposed\n"
            "  controls: Service is absent\n"
            "  confidence: High\n"
            "  best_model: openrouter:minimax/minimax-m3\n"
            "```\n"
        ),
        output="yaml",
    )

    result.write(output)

    assert list(output.rows()) == [
        {
            "cve_id": "CVE-2026-4035",
            "image": "image-one;image-two",
            "classification": "False positive",
            "rationale": "Package is present but service is not exposed",
            "controls": "Service is absent",
            "confidence": "High",
            "best_model": "openrouter:minimax/minimax-m3",
        }
    ]


def test_comparison_result_write_requires_yaml_output(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    output = database.create_table("output", OUTPUT_TABLE_SCHEMA)
    result = ComparisonResult(
        cve_id="CVE-2026-1234",
        prompt="Assess CVE-2026-1234",
        responses={},
        comparison_prompt="Compare analyses for CVE-2026-1234",
        comparison="| Image | Classification |",
        output="markdown",
    )

    with pytest.raises(ValueError, match="only supports yaml"):
        result.write(output)


def test_safe_save_returns_true_when_yaml_is_valid(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    output = database.create_table("output", OUTPUT_TABLE_SCHEMA)
    result = ComparisonResult(
        cve_id="CVE-2026-1234",
        prompt="Assess CVE-2026-1234",
        responses={},
        comparison_prompt="Compare model responses",
        comparison=(
            "- cve_id: CVE-2026-1234\n"
            "  image: ubuntu:latest\n"
            "  classification: Mitigated\n"
            "  rationale: Patched\n"
            "  controls: Network policy\n"
            "  confidence: High\n"
            "  best_model: first\n"
        ),
        output="yaml",
    )

    assert safe_save(result, FakeListChatModel(responses=[]), output) is True
    assert len(list(output.rows())) == 1


def test_safe_save_repairs_invalid_yaml_and_saves_retry(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    output = database.create_table("output", OUTPUT_TABLE_SCHEMA)
    result = ComparisonResult(
        cve_id="CVE-2026-1234",
        prompt="Assess CVE-2026-1234",
        responses={},
        comparison_prompt="Compare model responses",
        comparison="not valid YAML as a row",
        output="yaml",
    )
    judge = FakeListChatModel(
        responses=[
            "- cve_id: CVE-2026-1234\n"
            "  image: ubuntu:latest\n"
            "  classification: Mitigated\n"
            "  rationale: Patched\n"
            "  controls: Network policy\n"
            "  confidence: High\n"
            "  best_model: first\n"
        ]
    )

    assert safe_save(result, judge, output, count_max=2) is True
    assert len(list(output.rows())) == 1


def test_safe_save_returns_false_without_output_table() -> None:
    result = ComparisonResult(
        cve_id="CVE-2026-1234",
        prompt="Assess CVE-2026-1234",
        responses={},
        comparison_prompt="Compare model responses",
        comparison="- image: ubuntu:latest",
        output="yaml",
    )

    assert safe_save(result, FakeListChatModel(responses=[]), None) is False


def test_safe_save_returns_false_when_repair_attempt_is_exhausted(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    output = database.create_table("output", OUTPUT_TABLE_SCHEMA)
    result = ComparisonResult(
        cve_id="CVE-2026-1234",
        prompt="Assess CVE-2026-1234",
        responses={},
        comparison_prompt="Compare model responses",
        comparison="not valid YAML as a row",
        output="yaml",
    )
    judge = FakeListChatModel(responses=["still not a YAML list"])

    assert safe_save(result, judge, output, count_max=1) is False
    assert list(output.rows()) == []


@pytest.mark.parametrize(
    ("valid_initial", "valid_repair", "count_max", "has_table", "success", "repair_count"),
    [
        (True, True, 3, True, True, 0),
        (False, True, 2, True, True, 1),
        (False, True, 1, True, False, 0),
        (False, False, 3, True, False, 2),
        (False, True, 3, False, False, 0),
        (True, True, 0, True, False, 0),
        (True, True, -1, True, False, 0),
    ],
)
def test_asafe_save_awaits_only_model_repairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    valid_initial: bool,
    valid_repair: bool,
    count_max: int,
    has_table: bool,
    success: bool,
    repair_count: int,
) -> None:
    output = SQLiteClient(tmp_path).get_database("triage").create_table(
        "output", OUTPUT_TABLE_SCHEMA
    )
    valid_yaml = (
        "- image: ubuntu:latest\n"
        "  classification: Mitigated\n"
        "  rationale: Patched\n"
        "  controls: Network policy\n"
        "  confidence: High\n"
        "  best_model: first\n"
    )
    result = ComparisonResult(
        cve_id="CVE-2026-1234",
        prompt="Assess CVE-2026-1234",
        responses={},
        comparison_prompt="Compare model responses",
        comparison=valid_yaml if valid_initial else "not a YAML list",
        output="yaml",
    )
    async_invoke = AsyncMock(
        return_value=AIMessage(content=valid_yaml if valid_repair else "still not a YAML list")
    )
    sync_invoke = Mock(side_effect=AssertionError("Synchronous model call in async save"))
    monkeypatch.setattr(FakeListChatModel, "ainvoke", async_invoke)
    monkeypatch.setattr(FakeListChatModel, "invoke", sync_invoke)
    judge = FakeListChatModel(responses=[])

    assert asyncio.run(
        asafe_save(result, judge, output if has_table else None, count_max=count_max)
    ) is success
    assert async_invoke.await_count == repair_count
    sync_invoke.assert_not_called()
    assert len(list(output.rows())) == int(success)
