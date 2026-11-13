import asyncio
from itertools import batched
from logging import getLogger

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from cves_autotriager.llm import (
    OUTPUT_TABLE_SCHEMA,
    ComparisonResult,
    ModelComparisonChain,
    asafe_save,
)
from cves_autotriager.logging_utils import config_from_yaml
from cves_autotriager.parser import ImageReference, NVDEnricher, TrivyReportParser
from cves_autotriager.prompt import CVEPromptBuilder
from cves_autotriager.storage import SchemaType, SQLiteClient

config_from_yaml()

logger = getLogger(__name__)
COUNT_MAX = 3

df = TrivyReportParser("./data/1.11-ubuntu2/").to_dataframe()

criticals = df[df["severity"] == "HIGH"]

cves_id = criticals["id"].unique().tolist()

client = SQLiteClient("./data/db")
db = client.get_database("cves_autotriager")

if "nvd" not in db.tables:
    schema: list[tuple[str, SchemaType]] = [
        ("id", str),
        ("nvd_description", str),
        ("nvd_severity", str),
    ]
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

output_df = pd.DataFrame(output_table.rows(),columns=[column for column, _ in output_table.schema])

async def process_cve(
    cve_id: str, enrichment_lock: asyncio.Lock
) -> ComparisonResult | None:
    logger.info("Processing CVE: %s", cve_id)
    selected_cves = criticals.loc[criticals["id"] == cve_id]

    images = {
        image.unpinned
        for image in ImageReference.parse_many(
            ";".join(selected_cves["image"].unique().tolist())
        )
    }
    processed_images = {
        image.unpinned
        for image_group in output_df.loc[output_df["cve_id"] == cve_id, "image"].dropna()
        for image in ImageReference.parse_many(str(image_group))
    }
    left_over = images.difference(processed_images)

    if not left_over:
        logger.info("No new images to process for CVE: %s", cve_id)
        return None

    unprocessed_image_names = {
        image_name
        for image_name in selected_cves["image"].unique().tolist()
        if (image := ImageReference.parse(image_name)) is not None
        and image.unpinned in left_over
    }
    selected_cves = selected_cves.loc[selected_cves["image"].isin(unprocessed_image_names)]

    async with enrichment_lock:
        enriched = await enricher.aenrich(selected_cves)
    prompt = CVEPromptBuilder().build(cve_id, enriched)
    result = await model_chain.ainvoke(prompt, cve_id)

    success = await asafe_save(result, judge, output_table, count_max=COUNT_MAX)

    if not success:
        logger.error(
            "Failed to store comparison output for CVE: %s after %d attempts.", cve_id, COUNT_MAX
        )
    
    return result


async def process_cves(
    batch_size: int,
) -> list[tuple[str, ComparisonResult | BaseException | None]]:
    enrichment_lock = asyncio.Lock()
    results: list[tuple[str, ComparisonResult | BaseException | None]] = []
    for batch in batched(cves_id, batch_size):
        logger.info("Processing batch of CVEs: %s", batch)
        batch_results = await asyncio.gather(
            *(process_cve(cve_id, enrichment_lock) for cve_id in batch),
            return_exceptions=True,
        )
        results.extend(zip(batch, batch_results, strict=True))
    return results

results = asyncio.run(process_cves(batch_size=10))

for cve_id, result in results:
    if result is None:
        continue
    if isinstance(result, BaseException):
        logger.error("Failed to process CVE: %s. Error: %s", cve_id, result)
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

