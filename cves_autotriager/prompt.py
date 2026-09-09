"""Build triage prompts from parsed CVE data."""

from __future__ import annotations

from importlib import resources
from string import Template

import pandas as pd


class CVEPromptBuilder:
    """Render a triage prompt for a CVE represented in a dataframe."""

    def __init__(self, template_name: str = "triage_prompt.txt") -> None:
        template_path = resources.files("cves_autotriager.resources").joinpath(template_name)
        self._template = Template(template_path.read_text(encoding="utf-8"))

    def build(self, cve_id: str, cves: pd.DataFrame) -> str:
        """Build a prompt from enriched records for ``cve_id``."""
        required_columns = {
            "id",
            "image",
            "nvd_description",
            "nvd_severity",
            "package",
            "severity",
            "version",
        }
        missing_columns = required_columns.difference(cves.columns)
        if missing_columns:
            columns = ", ".join(sorted(missing_columns))
            raise ValueError(f"CVE dataframe is missing required columns: {columns}")

        matching_cves = cves.loc[cves["id"] == cve_id]
        if matching_cves.empty:
            raise ValueError(f"CVE not found in dataframe: {cve_id}")

        images = matching_cves["image"].dropna().astype(str).drop_duplicates().tolist()
        if not images:
            raise ValueError(f"CVE has no affected images: {cve_id}")

        severity = self._single_value(matching_cves, "nvd_severity", cve_id)
        description = self._single_value(matching_cves, "nvd_description", cve_id)
        image_list = "\n".join(f"* {image}" for image in images)
        findings = matching_cves[["image", "package", "version", "severity"]].drop_duplicates()
        finding_rows = "\n".join(
            "| " + " | ".join(self._escape_markdown(value) for value in row) + " |"
            for row in findings.itertuples(index=False, name=None)
        )
        return self._template.substitute(
            cve_id=cve_id,
            severity=severity,
            description=description,
            image_list=image_list,
            finding_rows=finding_rows,
        ).strip()

    @staticmethod
    def _single_value(cves: pd.DataFrame, column: str, cve_id: str) -> str:
        values = cves[column].dropna().astype(str).drop_duplicates().tolist()
        if not values:
            raise ValueError(f"CVE has no {column}: {cve_id}")
        if len(values) > 1:
            raise ValueError(f"CVE has conflicting {column} values: {cve_id}")
        return values[0]

    @staticmethod
    def _escape_markdown(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")
