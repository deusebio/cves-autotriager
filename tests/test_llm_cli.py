import argparse

import pytest

from llm import _model_spec, build_parser


def test_model_spec_accepts_provider_and_model() -> None:
    assert _model_spec("openai:gpt-5.4-mini") == "openai:gpt-5.4-mini"


@pytest.mark.parametrize("value", ["openai", ":gpt-5.4-mini", "openai:"])
def test_model_spec_rejects_incomplete_value(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="provider:model"):
        _model_spec(value)


def test_parser_collects_model_identifiers() -> None:
    args = build_parser().parse_args(
        [
            "reports",
            "CVE-2026-1234",
            "--model",
            "openai:gpt-5.4-mini",
            "--model",
            "anthropic:claude-sonnet-4-6",
            "--judge-model",
            "openai:gpt-5.5",
        ]
    )

    assert args.model == ["openai:gpt-5.4-mini", "anthropic:claude-sonnet-4-6"]