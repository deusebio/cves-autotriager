"""Run and compare CVE triage responses from LangChain chat models."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda, RunnableParallel

from cves_autotriager.llm.base import (
    BaseModelComparisonChain,
    TimedResponse,
)
from cves_autotriager.storage import Database


class ModelComparisonChain(BaseModelComparisonChain):
    """Send one prompt to LangChain candidate models and compare their responses."""

    def __init__(
        self,
        models: Sequence[BaseChatModel],
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

        self._candidate_models = {
            self._resolve_model_name(model, f"candidate_{index}"): model
            for index, model in enumerate(models)
        }

        if len(self._candidate_models) != len(models):
            raise ValueError("Candidate model names must be unique")

        self._candidates = RunnableParallel(self._candidate_models)
        self._judge = judge
        self._database = database

        super().__init__(
            list(self._candidate_models.keys()),
            self._resolve_model_name(judge, "judge"),
            candidate_table=self._ensure_table(database, candidate_table_name)
            if database is not None
            else None,
            judge_table=self._ensure_table(database, judge_table_name)
            if database is not None
            else None,
            comparison_template_name=comparison_template_name,
            output_format=output_format,
        )

    async def _invoke_candidate_models(
        self, model_names: list[str], prompt: str
    ) -> dict[str, TimedResponse]:
        timed_models = {
            name: RunnableLambda(self._timed_invoke(self._candidate_models[name]))
            for name in model_names
        }
        raw_results = await asyncio.to_thread(RunnableParallel(timed_models).invoke, prompt)
        responses: dict[str, TimedResponse] = {}
        for name, result in raw_results.items():
            if "response" in result:
                responses[name] = TimedResponse(
                    response=result["response"],
                    execution_time=float(result["execution_time"]),
                )
            elif "error" in result:
                self.logger.error(f"Error invoking model {name}: {result['error']}")
        return responses

    async def _invoke_judge_model(self, prompt: str) -> TimedResponse:
        started = time.perf_counter()
        comparison = self._message_text(await asyncio.to_thread(self._judge.invoke, prompt))
        return TimedResponse(response=comparison, execution_time=time.perf_counter() - started)

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

    @staticmethod
    def _resolve_model_name(model: BaseChatModel, default: str) -> str:
        for attr in ("model_name", "model", "name"):
            value = getattr(model, attr, None)
            if isinstance(value, str) and value:
                return value
        return default

    @staticmethod
    def _message_text(message: object) -> str:
        if not isinstance(message, BaseMessage):
            raise TypeError(f"Expected a LangChain message, found {type(message).__name__}")
        return message.text
