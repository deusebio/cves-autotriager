"""Run and compare CVE triage responses from LangChain chat models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from string import Template

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableParallel


@dataclass(frozen=True)
class ComparisonResult:
    """Candidate model responses and the judge model's comparison."""

    prompt: str
    responses: dict[str, str]
    comparison: str


class ModelComparisonChain:
    """Send one prompt to candidate models and compare their responses."""

    def __init__(
        self,
        models: dict[str, BaseChatModel],
        judge: BaseChatModel,
        comparison_template_name: str = "comparison_prompt.txt",
    ) -> None:
        if len(models) < 2:
            raise ValueError("At least two candidate models are required")
        if len(models) != len(set(models)):
            raise ValueError("Candidate model names must be unique")

        template_path = resources.files("cves_autotriager.resources").joinpath(
            comparison_template_name
        )
        self._template = Template(template_path.read_text(encoding="utf-8"))
        self._candidates = RunnableParallel(models)
        self._judge = judge

    def invoke(self, prompt: str) -> ComparisonResult:
        """Generate candidate answers in parallel, then invoke the judge."""
        raw_responses = self._candidates.invoke(prompt)
        responses = {
            name: self._message_text(response) for name, response in raw_responses.items()
        }
        comparison_prompt = self._template.substitute(
            original_prompt=prompt,
            responses=json.dumps(responses, indent=2, sort_keys=True),
        )
        comparison = self._message_text(self._judge.invoke(comparison_prompt))
        return ComparisonResult(prompt, responses, comparison)

    @staticmethod
    def _message_text(message: object) -> str:
        if not isinstance(message, BaseMessage):
            raise TypeError(f"Expected a LangChain message, found {type(message).__name__}")
        return message.text