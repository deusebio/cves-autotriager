"""Run and compare CVE triage responses using the GitHub Copilot SDK."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Sequence
from importlib import import_module
from typing import Any

from cves_autotriager.llm.base import (
    BaseModelComparisonChain,
    TimedResponse,
)
from cves_autotriager.storage import Database


class CopilotModelComparisonChain(BaseModelComparisonChain):
    """Send prompts to Copilot models, cache responses, and judge the results."""

    def __init__(
        self,
        candidate_model_names: Sequence[str],
        judge_model_name: str,
        *,
        database: Database | None = None,
        candidate_table_name: str = "candidate_responses",
        judge_table_name: str = "judge_responses",
        comparison_template_name: str = "comparison_prompt.txt",
        output_format: str = "markdown",
    ) -> None:
        if len(candidate_model_names) < 2:
            raise ValueError("At least two candidate models are required")
        if len(candidate_model_names) != len(set(candidate_model_names)):
            raise ValueError("Candidate model names must be unique")

        super().__init__(
            list(candidate_model_names),
            judge_model_name,
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
        pending_results = await asyncio.gather(
            *(self._timed_invoke(model_name, prompt) for model_name in model_names)
        )
        return dict(zip(model_names, pending_results, strict=True))

    async def _invoke_judge_model(self, prompt: str) -> TimedResponse:
        return await self._timed_invoke(self._judge_model_name, prompt)

    async def _timed_invoke(self, model_name: str, prompt: str) -> TimedResponse:
        started = time.perf_counter()
        response = await self._invoke_copilot_model(model_name, prompt)
        return TimedResponse(response=response, execution_time=time.perf_counter() - started)

    async def _invoke_copilot_model(self, model_name: str, prompt: str) -> str:
        try:
            copilot = import_module("copilot")
            session_module = import_module("copilot.session")
            events_module = import_module("copilot.session_events")
        except ImportError as error:
            raise RuntimeError(
                "github-copilot-sdk is required to use CopilotModelComparisonChain"
            ) from error

        copilot_client = copilot.CopilotClient
        permission_handler = session_module.PermissionHandler
        assistant_message_data = events_module.AssistantMessageData
        session_idle_data = events_module.SessionIdleData

        async with copilot_client() as client:
            return await self._send_prompt(
                client=client,
                model_name=model_name,
                prompt=prompt,
                permission_handler=permission_handler,
                assistant_message_data=assistant_message_data,
                session_idle_data=session_idle_data,
            )

    @staticmethod
    async def _send_prompt(
        *,
        client: Any,
        model_name: str,
        prompt: str,
        permission_handler: Any,
        assistant_message_data: type[Any],
        session_idle_data: type[Any],
    ) -> str:
        done = asyncio.Event()
        responses: list[str] = []

        async with await client.create_session(
            on_permission_request=permission_handler.approve_all,
            model=model_name,
        ) as session:

            def on_event(event: Any) -> None:
                data = event.data
                if isinstance(data, assistant_message_data):
                    responses.append(str(data.content))
                elif isinstance(data, session_idle_data):
                    done.set()

            session.on(on_event)
            await session.send(prompt)
            await done.wait()

        return "\n".join(responses).strip()
