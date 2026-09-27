from importlib import resources
from string import Template

from logging import getLogger
import pandas as pd

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from cves_autotriager.llm import OUTPUT_TABLE_SCHEMA, ModelComparisonChain, ComparisonResult, safe_save
from cves_autotriager.logging_utils import config_from_yaml
from cves_autotriager.parser import ImageReference, NVDEnricher, TrivyReportParser
from cves_autotriager.prompt import CVEPromptBuilder
from cves_autotriager.storage import SQLiteClient

config_from_yaml()

logger = getLogger(__name__)

df = TrivyReportParser("./data/1.11-ubuntu2/").to_dataframe()

criticals = df[df["severity"] == "CRITICAL"]

cves_id = criticals["id"].unique().tolist()

client = SQLiteClient("./data/db")
db = client.get_database("cves_autotriager")

if "nvd" not in db.tables:
    schema = [("id", str), ("nvd_description", str), ("nvd_severity", str)]
    nvd = db.create_table("nvd", schema)
else:
    nvd = db.get_table("nvd")

output_table = (
    db.get_table("output")
    if "output" in db.tables
    else db.create_table("output", OUTPUT_TABLE_SCHEMA)
)

enricher = NVDEnricher(nvd)

model_names = [
    "openrouter:deepseek/deepseek-v4-flash-0731",
    "openrouter:z-ai/glm-5.3-flash",
    # "openrouter:google/gemini-3.8-flash",
    "openrouter:minimax/minimax-m3",
]

judge_model_name = "openrouter:deepseek/deepseek-v4-flash-0731"

models: dict[str, BaseChatModel] = {model: init_chat_model(model) for model in model_names}

judge = init_chat_model(judge_model_name)

model_chain = ModelComparisonChain(models, judge, database=db, output_format="yaml")

cves_id = criticals["id"].unique().tolist()

output_df = pd.DataFrame(output_table.rows())

for cve_id in cves_id:

    logger.info("Processing CVE: %s", cve_id)
    selected_cves = criticals.loc[criticals["id"] == cve_id]

    images = {
        ImageReference.parse(image).unpinned 
        for image in selected_cves["image"].unique().tolist()
        }

    left_over = images.difference([
        ImageReference.parse(image).unpinned 
        for image in output_df[output_df["cve_id"] == cve_id]["image"].unique().tolist()
    ])

    if not left_over:
        logger.info("No new images to process for CVE: %s", cve_id)
        continue

    try:
        enriched = enricher.enrich(selected_cves)
        prompt = CVEPromptBuilder().build(cve_id, enriched)

        result = model_chain.invoke(prompt, cve_id)
    except Exception as e:
        logger.error("Failed to process CVE: %s. Error: %s", cve_id, e)
        continue

    if False:
        print()
        print("Comparison result:")
        for model_name, response in result.responses.items():
            print(f"============= {model_name} ===========================")
            print(response)
            print()  # Add a blank line for better readability between model responses

        print("============= Comparison ===========================")
        print(result.comparison)
        print()

    success = safe_save(result, judge, output_table, count_max=3)

    if not success:
        logger.error("Failed to store comparison output for CVE: %s after %d attempts.", cve_id, COUNT_MAX)

