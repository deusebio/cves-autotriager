from importlib import resources
from string import Template

from logging import getLogger

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from cves_autotriager.llm import OUTPUT_TABLE_SCHEMA, ModelComparisonChain, ComparisonResult
from cves_autotriager.logging_utils import config_from_yaml
from cves_autotriager.parser import NVDEnricher, TrivyReportParser
from cves_autotriager.prompt import CVEPromptBuilder
from cves_autotriager.storage import SQLiteClient

config_from_yaml()

logger = getLogger(__name__)

df = TrivyReportParser("./data/1.11-ubuntu2/").to_dataframe()

criticals = df[df["severity"] == "CRITICAL"]

cves_id = criticals["id"].unique().tolist()

# logger.info("List of critical CVEs:\n%s", "\n".join(cves_id))
# cve_id = input("Enter the CVE ID: ")
# selected_cves = criticals.loc[criticals["id"] == cve_id]

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

# enriched = enricher.enrich(selected_cves)

# prompt = CVEPromptBuilder().build(cve_id, enriched)

# logger.info("Prompt for CVE:\n%s", prompt)

# dry_run = input("Do you want to run the analysis? (yes/no): ")

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

cves_id = [
    "CVE-2026-48746", "CVE-2025-15379", "CVE-2026-0545", "CVE-2026-2635", "CVE-2026-4035", 
    "CVE-2024-41110", "CVE-2023-49569", "CVE-2022-1996", "CVE-2023-25668", "CVE-2017-7658", 
    "CVE-2023-50447", "CVE-2025-32434", "CVE-2024-8986", "CVE-2026-13221", "CVE-2026-57433"
]

COUNT_MAX=3
template_path = resources.files("cves_autotriager.resources").joinpath(
    "error_handler.txt"
)


# if dry_run.lower() == "yes":
for cve_id in cves_id:

    logger.info("Processing CVE: %s", cve_id)
    selected_cves = criticals.loc[criticals["id"] == cve_id]

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

    success = False
    count = 0
    while (not success) and (count<COUNT_MAX):
        count += 1
        try:
            result.write(output_table)
            logger.info("Stored comparison output rows in table 'output' for CVE: %s", cve_id)
            success = True
        except Exception as e:
            logger.error(f"[{count}/{COUNT_MAX}] Failed to process CVE: {cve_id}. Error: {e}")

            fix_prompt = Template(template_path.read_text(encoding="utf-8")).substitute(
                prompt=result.comparison_prompt,
                response=result.comparison,
                exception=str(e),
            )
            logger.info(f"{fix_prompt}")
            new_response = judge.invoke(fix_prompt).text
            logger.info("Received new response from judge for CVE: %s", cve_id)
            logger.info(f"{new_response}")
            result = ComparisonResult(
                cve_id=result.cve_id,
                prompt = result.prompt,
                responses=result.responses,
                comparison_prompt=fix_prompt,
                comparison=new_response,
                output=result.output,
            )

    if not success:
        logger.error("Failed to store comparison output for CVE: %s after %d attempts.", cve_id, COUNT_MAX)
