import pandas as pd
import pytest

from cves_autotriager.prompt import CVEPromptBuilder


def test_builder_populates_cve_and_unique_images() -> None:
    cves = pd.DataFrame(
        [
            {
                "id": "CVE-2026-1234",
                "image": "image-one:1.0",
                "package": "openssl",
                "version": "3.0.1",
                "severity": "HIGH",
                "nvd_severity": "HIGH",
                "nvd_description": "Example vulnerability",
            },
            {
                "id": "CVE-2026-1234",
                "image": "image-one:1.0",
                "package": "openssl",
                "version": "3.0.1",
                "severity": "HIGH",
                "nvd_severity": "HIGH",
                "nvd_description": "Example vulnerability",
            },
            {
                "id": "CVE-2026-1234",
                "image": "image-two:2.0",
                "package": "libssl3",
                "version": "3.0.2",
                "severity": "HIGH",
                "nvd_severity": "HIGH",
                "nvd_description": "Example vulnerability",
            },
            {
                "id": "CVE-2026-9999",
                "image": "other-image:1.0",
                "package": "other-package",
                "version": "1.0",
                "severity": "LOW",
                "nvd_severity": "LOW",
                "nvd_description": "Other vulnerability",
            },
        ]
    )

    prompt = CVEPromptBuilder().build("CVE-2026-1234", cves)

    assert "CVE ID: CVE-2026-1234" in prompt
    assert "NVD Severity: HIGH" in prompt
    assert "NVD Description: Example vulnerability" in prompt
    assert "* image-one:1.0\n* image-two:2.0" in prompt
    assert "| image-one:1.0 | openssl | 3.0.1 | HIGH |" in prompt
    assert "| image-two:2.0 | libssl3 | 3.0.2 | HIGH |" in prompt
    assert "**False positive**" in prompt
    assert "**Mitigated**" in prompt
    assert "**Exploitable**" in prompt
    assert "Do not assume that package presence alone proves exploitability" in prompt
    assert "other-image:1.0" not in prompt


def test_builder_rejects_unknown_cve() -> None:
    cves = pd.DataFrame(
        [
            {
                "id": "CVE-2026-1234",
                "image": "image-one:1.0",
                "package": "openssl",
                "version": "3.0.1",
                "severity": "HIGH",
                "nvd_severity": "HIGH",
                "nvd_description": "Example vulnerability",
            }
        ]
    )

    with pytest.raises(ValueError, match="CVE not found"):
        CVEPromptBuilder().build("CVE-2026-9999", cves)


def test_builder_requires_id_column() -> None:
    cves = pd.DataFrame(
        [
            {
                "image": "image-one:1.0",
                "package": "openssl",
                "version": "3.0.1",
                "severity": "HIGH",
                "nvd_severity": "HIGH",
                "nvd_description": "Example vulnerability",
            }
        ]
    )

    with pytest.raises(ValueError, match="missing required columns: id"):
        CVEPromptBuilder().build("CVE-2026-1234", cves)


def test_builder_rejects_missing_nvd_details() -> None:
    cves = pd.DataFrame(
        [
            {
                "id": "CVE-2026-1234",
                "image": "image-one:1.0",
                "package": "openssl",
                "version": "3.0.1",
                "severity": "HIGH",
                "nvd_severity": None,
                "nvd_description": "Example vulnerability",
            }
        ]
    )

    with pytest.raises(ValueError, match="no nvd_severity"):
        CVEPromptBuilder().build("CVE-2026-1234", cves)
