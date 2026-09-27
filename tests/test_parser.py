import json
from pathlib import Path

from cves_autotriager.parser import CVE_COLUMNS, ImageReference, TrivyReportParser


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


def test_image_reference_parses_single_component_docker_image() -> None:
    reference = ImageReference.parse("ubuntu:latest")

    assert reference == ImageReference("docker.io", "ubuntu", "latest")
    assert reference is not None
    assert reference.unpinned == ImageReference("docker.io", "ubuntu")


def test_image_reference_parses_registry_and_digest() -> None:
    reference = ImageReference.parse("ghcr.io/owner/image@sha256:abc123")

    assert reference == ImageReference("ghcr.io", "owner/image", "abc123")


def test_image_reference_parses_semicolon_group() -> None:
    references = ImageReference.parse_many(
        "docker.io/library/ubuntu:latest; ghcr.io/owner/image:v1"
    )

    assert references == {
        ImageReference("docker.io", "library/ubuntu", "latest"),
        ImageReference("ghcr.io", "owner/image", "v1"),
    }


def test_image_reference_returns_none_for_invalid_reference() -> None:
    assert ImageReference.parse("Not An Image") is None
