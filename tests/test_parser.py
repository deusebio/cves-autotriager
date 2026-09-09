import json
from pathlib import Path

from cves_autotriager.parser import CVE_COLUMNS, TrivyReportParser


def test_parser_extracts_cves_into_dataframe(tmp_path: Path) -> None:
    report_directory = tmp_path / "canonical"
    report_directory.mkdir()
    report = {
        "ArtifactName": "charmedkubeflow/api-server:1.11",
        "Results": [
            {
                "Vulnerabilities": [
                    {
                        "VulnerabilityID": "CVE-2026-1234",
                        "Severity": "HIGH",
                        "PkgName": "openssl",
                        "InstalledVersion": "3.0.1",
                    }
                ]
            }
        ],
    }
    (report_directory / "api_scan.json").write_text(json.dumps(report), encoding="utf-8")

    dataframe = TrivyReportParser(tmp_path).to_dataframe()

    assert list(dataframe.columns) == CVE_COLUMNS
    assert dataframe.to_dict(orient="records") == [
        {
            "id": "CVE-2026-1234",
            "severity": "HIGH",
            "package": "openssl",
            "version": "3.0.1",
            "image": "charmedkubeflow/api-server:1.11",
            "folder": "canonical",
            "file": "api",
        }
    ]


def test_parser_returns_empty_dataframe_for_report_without_cves(tmp_path: Path) -> None:
    report = {"ArtifactName": "example/image:latest", "Results": []}
    (tmp_path / "clean.json").write_text(json.dumps(report), encoding="utf-8")

    dataframe = TrivyReportParser(tmp_path).to_dataframe()

    assert dataframe.empty
    assert list(dataframe.columns) == CVE_COLUMNS