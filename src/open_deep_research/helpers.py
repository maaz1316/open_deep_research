from enum import IntEnum
import logging

import aiohttp
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from typing_extensions import Dict, List, Optional

from infras.secrets.constants import SecretKeys
from libs.secrets.secrets import Secrets


def get_logger():
    """Returns the configured logger."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    return logger


class ResponseStatus(IntEnum):
    start = 1
    industry_and_competitors = 2
    shortlisting = 3
    downloading_source_file = 4
    logs_start = 5
    logs_stream = 6
    report_start = 7
    report_stream = 8
    references = 9
    complete = 10
    exception = 11
    summary = 12


class PatchRequest(BaseModel):
    id: str
    status: int
    industry: Optional[str] = ""
    competitors: Optional[List[str]] = []
    logs: Optional[Dict] = None
    reportContent: Optional[str] = None
    summaryContent: Optional[str] = None
    references: Optional[List[str]] = []
    fullname: Optional[str] = ""
    ticker: Optional[str] = ""
    first_name: Optional[str] = ""
    source: Optional[str] = ""
    language: Optional[str] = ""
    corp_code: Optional[str] = ""
    company_url: Optional[str] = ""


BASE_URL = Secrets.get(SecretKeys.BASE_URL)
patch_request_url = f"""{BASE_URL}/api/report/im-workflow"""

# Configure logger
logger = get_logger()


async def patch_url_function_progress(url, response_data: PatchRequest):
    try:
        logger.info(url)
        async with aiohttp.ClientSession() as session:
            async with session.patch(url, json=response_data.model_dump()) as response:
                if response.status != 200:
                    logger.info(f"Failed with status code: {response.status}")
                    logger.info(await response.text())
                    raise Exception(f"Patch request failed with status code: {response.status}")
        logger.info(response_data.model_dump_json())
    except Exception as e:
        logger.info(f"An error occurred while patching the URL: URL :{url}\nRESPONSE: {response_data.model_dump_json()}\nERROR: {str(e)}")
        raise Exception("patch_url_function_progress --> " + str(e))


async def generate_summary_on_report(report: str) -> str:
    try:
        llm = ChatOpenAI(
            model="gpt-4.1-nano", temperature=0, api_key=Secrets.get(SecretKeys.OPENAI_API_KEY)
        )  # Assuming gpt-4.1-nano is available, adjust if needed

        # Define the prompt for the LLM
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an industry expert with over 10 years of investment banking experience."
                        "You are supporting an M&A transaction in a given sector."
                        "Provide an expert insight in one paragraph, no longer than 100 words."
                        "Keep the points succinct yet well grounded."
                    ),
                ),
                ("human", "{text}"),
            ]
        )

        # Use Langchain's with_structured_output to ensure Pydantic model adherence

        # Invoke the LLM with the text
        summary_report_chain = prompt | llm
        summary_report = await summary_report_chain.ainvoke({"text": report})

        return summary_report.content
    except Exception:
        return []


async def generate_company_summary(company_url):
    client = AsyncOpenAI(api_key=Secrets.get(SecretKeys.OPENAI_API_KEY))
    try:
        response = await client.responses.parse(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Generate one paragraph summary of this company {company_url} using web."},
            ],
            tools=[
                {"type": "web_search"},
            ],
        )

        return response
    except Exception as e:
        logger.info(f"Error generating company summary:{e}")
        return ""


def company_translation(company_name):
    class CompanyName(BaseModel):
        company_name: str = Field(description="english name of the company")

    client = OpenAI(api_key=Secrets.get(SecretKeys.OPENAI_API_KEY))

    try:
        response = client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {
                    "role": "system",
                    "content": (
                        "Translate the name of the company into English. "
                        "Return the company name if it is already in English; "
                        "otherwise, return the translated company name."
                    ),
                },
                {"role": "user", "content": f"Company Name: {company_name}"},
            ],
            text_format=CompanyName,
        )

        return response.output_parsed.company_name

    except Exception as e:
        logger.info("Translation error:", e)
        return company_name
