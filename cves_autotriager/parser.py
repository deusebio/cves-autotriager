"""Parse CVEs from Trivy JSON reports."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from cves_autotriager.logging_utils import WithLogging
from cves_autotriager.storage import DataType, Table

CVE_COLUMNS = ["id", "severity", "package", "version", "image", "folder", "file"]


class TrivyReportParser:
    """Discover Trivy reports and extract their vulnerabilities."""

    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)

    def report_files(self) -> Iterator[Path]:
        """Yield JSON reports below the configured root in stable order."""
        if not self.root_path.is_dir():
            raise NotADirectoryError(self.root_path)
        yield from sorted(self.root_path.rglob("*.json"))

    def records(self) -> Iterator[dict[str, str]]:
        """Yield one normalized record for each vulnerability."""
        for report_path in self.report_files():
            yield from self._parse_report(report_path)

    def to_dataframe(self) -> pd.DataFrame:
        """Return all vulnerabilities as a dataframe with stable columns."""
        return pd.DataFrame.from_records(self.records(), columns=CVE_COLUMNS)

    def _parse_report(self, report_path: Path) -> Iterator[dict[str, str]]:
        try:
            with report_path.open(encoding="utf-8") as report_file:
                report: Any = json.load(report_file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read Trivy report {report_path}") from error

        if not isinstance(report, dict):
            raise ValueError(f"Trivy report must contain a JSON object: {report_path}")

        artifact_name = report.get("ArtifactName")
        if not isinstance(artifact_name, str):
            raise ValueError(f"Trivy report is missing ArtifactName: {report_path}")

        results = report.get("Results", [])
        if not isinstance(results, list):
            raise ValueError(f"Trivy report Results must be a list: {report_path}")

        relative_parent = report_path.parent.relative_to(self.root_path)
        report_name = report_path.stem.removesuffix("_scan")
        for result in results:
            if not isinstance(result, dict):
                continue
            vulnerabilities = result.get("Vulnerabilities", [])
            if not isinstance(vulnerabilities, list):
                continue
            for vulnerability in vulnerabilities:
                if not isinstance(vulnerability, dict):
                    continue
                try:
                    yield {
                        "id": str(vulnerability["VulnerabilityID"]),
                        "severity": str(vulnerability["Severity"]),
                        "package": str(vulnerability["PkgName"]),
                        "version": str(vulnerability["InstalledVersion"]),
                        "image": artifact_name,
                        "folder": str(relative_parent),
                        "file": report_name,
                    }
                except KeyError as error:
                    raise ValueError(
                        f"Vulnerability in {report_path} is missing {error.args[0]}"
                    ) from error


class NVDEnricher(WithLogging):
    """Enrich CVEs with NVD data cached in a SQLite table."""

    def __init__(
        self,
        table: Table,
        base_url: str = "https://services.nvd.nist.gov/rest/json/cves/2.0",
        request_delay: float = 6,
    ) -> None:
        required_columns = {"id", "nvd_description", "nvd_severity"}
        table_columns = {name for name, _ in table.schema}
        if not required_columns.issubset(table_columns):
            missing = ", ".join(sorted(required_columns - table_columns))
            raise ValueError(f"NVD cache table is missing columns: {missing}")
        self.table = table
        self.base_url = base_url
        self.request_delay = request_delay

    def _get_cve_details(self, cve_id: str) -> dict[str, DataType]:

        if self.table and (cached := self.table.find_one("id", cve_id)) is not None:
            self.logger.debug(f"Found CVE with id {cve_id}")
            return cached

        self.logger.debug(f"CVE with id {cve_id} not found in cache")
        details = self._fetch_cve_details(cve_id)

        if "error" in details:
            self.logger.warning(f"There has been an error fetching CVE details for {cve_id}: {details['error']}")
            return details

        if self.table:
            self.table.insert([details[column] for column, _ in self.table.schema])
        return details

    def _fetch_cve_details(self, cve_id: str) -> dict[str, DataType]:
        self.logger.info(f"Retrieving information for CVE with id {cve_id}")

        params = {"cveId": cve_id}
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            vuln = data.get("vulnerabilities", [{}])[0].get("cve", {})
            return {
                "id": cve_id,
                "nvd_description": vuln.get("descriptions", [{}])[0].get("value"),
                "nvd_severity": vuln.get("metrics", {})
                    .get("cvssMetricV31", [{}])[0]
                    .get("cvssData", {})
                    .get("baseSeverity"),
            }
        except requests.RequestException as error:
            return {"id": cve_id, "error": str(error)}
        finally:
            time.sleep(self.request_delay)

    def enrich(self, cves: pd.DataFrame) -> pd.DataFrame:
        if "id" not in cves.columns:
            raise ValueError("CVE dataframe must contain an 'id' column")
        enriched_cves: list[dict[str, Any]] = []
        for _, row in cves.iterrows():
            _id = row.pop("id")
            self.logger.debug(f"Processing CVE with id {_id}")
            cve_details = self._get_cve_details(str(_id))
            enriched_cves.append({"id": _id, **cve_details, **row})
        return pd.DataFrame(enriched_cves)