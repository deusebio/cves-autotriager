from importlib import resources

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from cves_autotriager.llm import ModelComparisonChain


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


def test_comparison_prompt_allows_indeterminate_without_majority_vote() -> None:
    template = (
        resources.files("cves_autotriager.resources")
        .joinpath("comparison_prompt.txt")
        .read_text(encoding="utf-8")
    )

    assert "**Indeterminate**" in template
    assert "do not resolve uncertainty by majority vote" in template
