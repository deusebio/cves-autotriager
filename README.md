# cves-autotriager

Automated triage tooling for CVEs.

## Requirements

- Python >= 3.10
- [Poetry](https://python-poetry.org/) >= 2.0

## Setup

```bash
poetry install
```

## Usage

Parse a directory of Trivy JSON reports:

```python
from cves_autotriager import CVEPromptBuilder, TrivyReportParser

cves = TrivyReportParser("trivy-reports/canonical").to_dataframe()
```

Initialize a local database to store informations:

```python
from cves_autotriager import SQLiteClient

client = SQLiteClient("data/db")
database = client.get_database("cves_autotriager")
```

Enrich the CVEs with NVD information. You can use a table to cache responses from NVD server:

```python
from cves_autotriager import NVDEnricher

nvd_cache = database.create_table(
	"nvd_cache",
	[("id", str), ("nvd_description", str), ("nvd_severity", str)],
)
enriched_cves = NVDEnricher(nvd_cache).enrich(cves)
```

Existing cache entries are returned without an NVD request. Successful new
responses are inserted into the table before enrichment continues.

Compare CVE analyses from multiple LangChain models and aggregate the results using a judge model:

```python
from langchain.chat_models import init_chat_model

from cves_autotriager import CVEPromptBuilder, NVDEnricher, SQLiteClient, TrivyReportParser
from cves_autotriager.llm.langchain import OUTPUT_TABLE_SCHEMA, ModelComparisonChain

cve_id = "..."

prompt = CVEPromptBuilder().build(cve_id, enriched_cves)

model_names = [
	"openrouter:deepseek/deepseek-v4-flash-0731",
	"openrouter:z-ai/glm-5.3-flash",
	"openrouter:minimax/minimax-m3",
]
models = [init_chat_model(name) for name in model_names]
judge = init_chat_model("openrouter:deepseek/deepseek-v4-flash-0731")

result = ModelComparisonChain(
	models,
	judge,
	database=database,
	output_format="yaml",
).invoke(prompt, cve_id)
```

Candidate and judge model responses are cached in SQLite when a database is
provided. If the same prompt is submitted again for the same CVE and model name,
the cached response is reused instead of calling the model provider again. 

When the output of the comparison model is `yaml`, the summary of assessment can also be then stored in a table using `ComparisonResult.write(...)`:

```python
from cves_autotriager.llm.langchain import OUTPUT_TABLE_SCHEMA

output_table = (
	database.get_table("output")
	if "output" in database.tables
	else database.create_table("output", OUTPUT_TABLE_SCHEMA)
)
result.write(output_table)
```

### Customization

The editable prompt template is in `cves_autotriager/resources/triage_prompt.txt` and `cves_autotriager/resources/comparison_prompt.txt`.

### Example

The repo-root `main.py` script is a smoke-test workflow that parses local Trivy reports,
enriches the selected CVE with NVD data, sends the triage prompt to multiple
LangChain models, asks a judge model to compare their answers, and stores the
YAML judge output in SQLite.

Run it with:

```bash
poetry run python main.py
```

Model credentials are read by the relevant LangChain providers from environment
variables. For OpenRouter-backed models, configure the provider credentials
before running the script.


## Development

Run the test suite:

```bash
poetry run pytest
```

Lint and format:

```bash
poetry run ruff check .
poetry run ruff format .
```

Type-check:

```bash
poetry run mypy cves_autotriager
```

GitHub Actions runs the same lint, type-check, and unit-test commands in three
parallel jobs on every push and pull request.
