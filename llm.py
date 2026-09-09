"""Run CVE triage through multiple models and compare their answers."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from cves_autotriager.llm import ModelComparisonChain
from cves_autotriager.logging_utils import config_from_file
from cves_autotriager.parser import NVDEnricher, TrivyReportParser
from cves_autotriager.prompt import CVEPromptBuilder
from cves_autotriager.storage import SQLiteClient


def _model_spec(value: str) -> str:
    provider, separator, model = value.partition(":")
    if not separator or not provider or not model:
        raise argparse.ArgumentTypeError("model must use provider:model")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", help="Directory containing Trivy JSON reports")
    parser.add_argument("cve_id", help="CVE identifier to triage")
    parser.add_argument(
        "--cache-directory",
        default="data/db",
        help="Directory containing the local NVD cache database",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        type=_model_spec,
        metavar="provider:model",
        help="Candidate model; provide this option at least twice",
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help="LangChain judge model identifier, for example openai:gpt-5.4-mini",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_from_file()

    model_specs: list[str] = args.model
    if len(model_specs) < 2:
        build_parser().error("--model must be provided at least twice")
    if len(set(model_specs)) != len(model_specs):
        build_parser().error("candidate models must be unique")

    cves = TrivyReportParser(args.reports).to_dataframe()
    selected_cves = cves.loc[cves["id"] == args.cve_id]
    if selected_cves.empty:
        build_parser().error(f"CVE not found in reports: {args.cve_id}")

    database = SQLiteClient(args.cache_directory).get_database("cves_autotriager")
    if "nvd" in database.tables:
        nvd_cache = database.get_table("nvd")
    else:
        nvd_cache = database.create_table(
            "nvd",
            [("id", str), ("nvd_description", str), ("nvd_severity", str)],
        )
    enriched_cves = NVDEnricher(nvd_cache).enrich(selected_cves)
    prompt = CVEPromptBuilder().build(args.cve_id, enriched_cves)
    models: dict[str, BaseChatModel] = {model: init_chat_model(model) for model in model_specs}
    judge = init_chat_model(args.judge_model)
    result = ModelComparisonChain(models, judge).invoke(prompt)
    print(
        json.dumps(
            {
                "prompt": result.prompt,
                "responses": result.responses,
                "comparison": result.comparison,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
