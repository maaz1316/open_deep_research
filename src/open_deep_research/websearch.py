import asyncio
from datetime import datetime
from io import BytesIO
import json

from html_to_markdown import convert_to_markdown
import httpx
from langchain_core.documents import Document
from llama_parse import LlamaParse
from pydantic import BaseModel
import requests
from typing_extensions import List, Optional

from infras.secrets.constants import SecretKeys
from libs.secrets.secrets import Secrets

from .helpers import get_logger

logger = get_logger()


async def perplexity_doc_search(company_url):

    class ReportUrl(BaseModel):
        urls: Optional[List[str]] = None

    url = "https://api.perplexity.ai/chat/completions"
    headers = {"Authorization": f"Bearer {Secrets.get(SecretKeys.PERPLEXITY_API_KEY)}", "Content-Type": "application/json"}

    payload = {
        "model": "sonar-pro",
        "messages": [
            {
                "role": "user",
                "content": f"""Find the annual filing report (annual report or 10-K equivalent)
            for the company of the year {datetime.today().year - 1} or {datetime.today().year} associated with the domain "{company_url}".
            Instructions:
            1. Search the official company website ({company_url}) for investor relations or financial filings.
            2. Return only the direct URL(s) to the report, preferably ending with .pdf.
            3. Share if multiple urls are available.
            4. Return only if the documents are available on {company_url} website.
            """,
            }
        ],
        "response_format": {"type": "json_schema", "json_schema": {"schema": ReportUrl.model_json_schema()}},
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(url, headers=headers, json=payload, timeout=60)
        data = r.json()

    return json.loads(data["choices"][0]["message"]["content"])["urls"]


async def fetch_and_parse(url: str):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    }

    parser = LlamaParse(
        api_key=Secrets.get(SecretKeys.LLAMA_CLOUD_API_KEY),
        result_type="markdown",
        verbose=True,
        high_res_ocr=True,
        language="en",
        disable_image_extraction=True,
        output_tables_as_HTML=True,
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="gemini-2.5-flash",
        vendor_multimodal_api_key=Secrets.get(SecretKeys.GEMINI_API_KEY),
    )

    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=120, headers=headers)

    if r.status_code == 200:
        file = BytesIO(r.content)
        docs = await parser.aparse(file, extra_info={"file_name": url})
        return docs
    return None


async def is_valid_pdf_url(url):
    if url.lower().endswith(".pdf"):
        return True


async def get_docs(company_url: str):
    pdf_urls = await perplexity_doc_search(company_url)
    if not pdf_urls:
        logger.info("********************No Docs found from web**************************")
        return []

    valid_urls = []
    for url in pdf_urls:
        if await is_valid_pdf_url(url):
            valid_urls.append(url)
        else:
            logger.info(f"Skipping invalid PDF URL: {url}")

    logger.info(f"Found Docs:{" ,".join(valid_urls)} ")

    results = await asyncio.gather(*(fetch_and_parse(url) for url in valid_urls))

    langchain_docs = []
    for idx, parsed_docs in enumerate(results):
        if not parsed_docs:
            continue
        json_doc = await parsed_docs.aget_json()
        for i, page in enumerate(json_doc["pages"]):
            page_content = convert_to_markdown(page["md"])
            langchain_doc = Document(
                page_content=page_content,
                metadata={"url": pdf_urls[idx], "page": f"page_{i + 1}", "document_type": parsed_docs.file_name},
            )
            langchain_docs.append(langchain_doc)

    return langchain_docs
