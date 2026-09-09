from pathlib import Path
from typing import Any

import pytest

from cves_autotriager.parser import NVDEnricher
from cves_autotriager.storage import SQLiteClient


class FakeResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {
            "vulnerabilities": [
                {
                    "cve": {
                        "descriptions": [{"value": "Example vulnerability"}],
                        "metrics": {
                            "cvssMetricV31": [
                                {"cvssData": {"baseSeverity": "HIGH"}}
                            ]
                        },
                    }
                }
            ]
        }


def test_get_cve_details_caches_nvd_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = SQLiteClient(tmp_path).get_database("triage").create_table(
        "nvd_cache",
        [("id", str), ("nvd_description", str), ("nvd_severity", str)],
    )
    request_count = 0

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal request_count
        request_count += 1
        return FakeResponse()

    monkeypatch.setattr("cves_autotriager.parser.requests.get", fake_get)
    enricher = NVDEnricher(table, request_delay=0)

    first = enricher._get_cve_details("CVE-2026-1234")
    second = enricher._get_cve_details("CVE-2026-1234")

    assert first == second == {
        "id": "CVE-2026-1234",
        "nvd_description": "Example vulnerability",
        "nvd_severity": "HIGH",
    }
    assert request_count == 1
    assert list(table.rows()) == [first]