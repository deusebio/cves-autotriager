from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest

from cves_autotriager.llm.copilot import CopilotModelComparisonChain
from cves_autotriager.storage import SQLiteClient

AsyncModelInvoker = Callable[[str, str], Awaitable[str]]


def make_mock(model_invoker: AsyncModelInvoker) -> type[CopilotModelComparisonChain]:
    class MockedCopilotModelComparisonChain(CopilotModelComparisonChain):
        async def _invoke_copilot_model(self, model_name: str, prompt: str) -> str:
            return await model_invoker(model_name, prompt)

    return MockedCopilotModelComparisonChain


def test_copilot_comparison_chain_invokes_candidates_and_judge(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    calls: list[tuple[str, str]] = []

    async def model_invoker(model_name: str, prompt: str) -> str:
        calls.append((model_name, prompt))
        if model_name == "judge-model":
            assert "first analysis" in prompt
            assert "second analysis" in prompt
            return "- image: image-one\n  classification: Mitigated\n"
        return {
            "candidate-one": "first analysis",
            "candidate-two": "second analysis",
        }[model_name]

    result = make_mock(model_invoker)(
        ["candidate-one", "candidate-two"],
        "judge-model",
        database=database,
        candidate_table_name="copilot_candidate_responses",
        judge_table_name="copilot_judge_responses",
        output_format="yaml",
    ).invoke("Assess CVE-2026-1234", cve_id="CVE-2026-1234")

    assert result.prompt == "Assess CVE-2026-1234"
    assert result.responses == {
        "candidate-one": "first analysis",
        "candidate-two": "second analysis",
    }
    assert result.comparison.startswith("- image: image-one")
    assert [model_name for model_name, _ in calls] == [
        "candidate-one",
        "candidate-two",
        "judge-model",
    ]


def test_copilot_comparison_chain_reuses_cached_responses(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    calls = 0

    async def model_invoker(model_name: str, prompt: str) -> str:
        nonlocal calls
        calls += 1
        return f"{model_name}: {prompt}"

    chain = make_mock(model_invoker)(
        ["candidate-one", "candidate-two"],
        "judge-model",
        database=database,
        candidate_table_name="copilot_candidate_responses",
        judge_table_name="copilot_judge_responses",
    )

    first = chain.invoke("Assess CVE-2026-1234", cve_id="CVE-2026-1234")
    second = chain.invoke("Assess CVE-2026-1234", cve_id="CVE-2026-1234")

    assert second.responses == first.responses
    assert second.comparison == first.comparison
    assert calls == 3
    candidate_responses = database.get_table("copilot_candidate_responses")
    judge_responses = database.get_table("copilot_judge_responses")
    assert len(list(candidate_responses.rows())) == 2
    assert len(list(judge_responses.rows())) == 1


def test_copilot_comparison_chain_rejects_wrong_table_schema(tmp_path: Path) -> None:
    database = SQLiteClient(tmp_path).get_database("triage")
    database.create_table("wrong", [("model_name", str)])

    with pytest.raises(ValueError, match="Response table schema mismatch"):
        CopilotModelComparisonChain(
            ["candidate-one", "candidate-two"],
            "judge-model",
            database=database,
            candidate_table_name="wrong",
            judge_table_name="wrong",
        )
