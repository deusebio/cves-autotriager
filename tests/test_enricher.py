import asyncio
from pathlib import Path
from typing import Any

import pandas as pd
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
    async_request_count = 0

    def fake_get(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal request_count
        request_count += 1
        return FakeResponse()

    async def fake_async_get(*args: Any, **kwargs: Any) -> FakeResponse:
        nonlocal async_request_count
        async_request_count += 1
        return FakeResponse()

    monkeypatch.setattr("cves_autotriager.parser.requests.get", fake_get)
    monkeypatch.setattr("cves_autotriager.parser.httpx.AsyncClient.get", fake_async_get)
    enricher = NVDEnricher(table, request_delay=0)

    sync_input = pd.DataFrame([{"id": "CVE-2026-1234", "package": "sync"}])
    first = enricher.enrich(sync_input)
    second = enricher.enrich(sync_input)

    async def get_async_details() -> pd.DataFrame:
        async_input = pd.DataFrame([{"id": "CVE-2026-5678", "package": "async"}])
        first = await enricher.aenrich(async_input)
        second = await enricher.aenrich(async_input)
        assert first.equals(second)
        return first

    async_enriched = asyncio.run(get_async_details())

    expected_sync = {
        "id": "CVE-2026-1234",
        "nvd_description": "Example vulnerability",
        "nvd_severity": "HIGH",
    }
    expected_async = {
        "id": "CVE-2026-5678",
        "nvd_description": "Example vulnerability",
        "nvd_severity": "HIGH",
        "package": "async",
    }
    assert first.iloc[0][["id", "nvd_description", "nvd_severity"]].to_dict() == expected_sync
    assert second.equals(first)
    assert async_enriched.iloc[0].to_dict() == expected_async
    assert request_count == async_request_count == 1
    assert len(list(table.rows())) == 2