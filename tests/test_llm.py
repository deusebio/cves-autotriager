from importlib import resources
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cves_autotriager.llm import OUTPUT_TABLE_SCHEMA, ComparisonResult, ModelComparisonChain
from cves_autotriager.storage import SQLiteClient


def test_comparison_chain_fans_out_prompt_and_invokes_judge() -> None:
    first = FakeListChatModel(responses=["first analysis"])
    second = FakeListChatModel(responses=["second analysis"])
    judge = FakeListChatModel(responses=["second is better supported"])

    result = ModelComparisonChain(
        {"first": first, "second": second},
        judge,
    ).invoke("Assess CVE-2026-1234")

    assert result.prompt == "Assess CVE-2026-1234"
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
    assert "For YAML output format" in template
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


def test_comparison_result_writes_yaml_output_to_table(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    output = database.create_table("output", OUTPUT_TABLE_SCHEMA)
    result = ComparisonResult(
        prompt="Assess CVE-2026-1234",
        responses={},
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
            "image": "ubuntu:1.11",
            "classification": "Mitigated",
            "rationale": "Patched package",
            "controls": "Network policy",
            "confidence": "High",
            "best_model": "first",
        }
    ]


def test_comparison_result_write_requires_yaml_output(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    output = database.create_table("output", OUTPUT_TABLE_SCHEMA)
    result = ComparisonResult(
        prompt="Assess CVE-2026-1234",
        responses={},
        comparison="| Image | Classification |",
        output="markdown",
    )

    with pytest.raises(ValueError, match="only supports yaml"):
        result.write(output)
