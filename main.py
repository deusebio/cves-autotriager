from logging import getLogger

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from cves_autotriager.llm import OUTPUT_TABLE_SCHEMA, ModelComparisonChain
from cves_autotriager.logging_utils import config_from_yaml
from cves_autotriager.parser import NVDEnricher, TrivyReportParser
from cves_autotriager.prompt import CVEPromptBuilder
from cves_autotriager.storage import SQLiteClient

config_from_yaml()

logger = getLogger(__name__)

df = TrivyReportParser("./data/1.11-ubuntu2/").to_dataframe()

criticals = df[df["severity"] == "CRITICAL"]

cves_id = criticals["id"].unique().tolist()

logger.info("List of critical CVEs:\n%s", "\n".join(cves_id))

cve_id = input("Enter the CVE ID: ")

selected_cves = criticals.loc[criticals["id"] == cve_id]


client = SQLiteClient("./data/db")
db = client.get_database("cves_autotriager")

if "nvd" not in db.tables:
    schema = [("id", str), ("nvd_description", str), ("nvd_severity", str)]
    nvd = db.create_table("nvd", schema)
else:
    nvd = db.get_table("nvd")

enricher = NVDEnricher(nvd)
enriched = enricher.enrich(selected_cves)

prompt = CVEPromptBuilder().build(cve_id, enriched)

logger.info("Prompt for CVE:\n%s", prompt)

dry_run = input("Do you want to run the analysis? (yes/no): ")

model_names = [
    "openrouter:deepseek/deepseek-v4-flash-0731",
    "openrouter:z-ai/glm-5.3-flash",
    # "openrouter:google/gemini-3.8-flash",
    "openrouter:minimax/minimax-m3",
]

judge_model_name = "openrouter:deepseek/deepseek-v4-flash-0731"

models: dict[str, BaseChatModel] = {model: init_chat_model(model) for model in model_names}

judge = init_chat_model(judge_model_name)

if dry_run.lower() == "yes":
    result = ModelComparisonChain(models, judge, database=db, output_format="yaml").invoke(
        prompt, cve_id
    )

    print()
    print("Comparison result:")
    for model_name, response in result.responses.items():
        print(f"============= {model_name} ===========================")
        print(response)
        print()  # Add a blank line for better readability between model responses

    print("============= Comparison ===========================")
    print(result.comparison)
    print()

    output_table = (
        db.get_table("output")
        if "output" in db.tables
        else db.create_table("output", OUTPUT_TABLE_SCHEMA)
    )
    result.write(output_table)
    logger.info("Stored comparison output rows in table 'output'")
