import pytest

from cves_autotriager import __version__
from cves_autotriager.cli import main


def test_main_returns_zero() -> None:
    assert main([]) == 0


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_repo_main_matches_current_cli_contract() -> None:
    import main

    parser = main.build_parser()
    args = parser.parse_args(
        [
            "data/reports",
            "CVE-2026-45623",
            "--model",
            "openai:gpt-5.4-mini",
            "--model",
            "anthropic:claude-sonnet-4",
            "--judge-model",
            "openai:gpt-5.5",
        ]
    )

    assert args.reports == "data/reports"
    assert args.cve_id == "CVE-2026-45623"
    assert args.model == ["openai:gpt-5.4-mini", "anthropic:claude-sonnet-4"]
    assert args.judge_model == "openai:gpt-5.5"
