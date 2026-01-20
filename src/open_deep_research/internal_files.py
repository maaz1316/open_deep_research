import asyncio

from html_to_markdown import convert_to_markdown
import httpx
from langchain_core.documents import Document
from llama_cloud_services import LlamaParse

from infras.secrets.constants import SecretKeys
from libs.secrets.secrets import Secrets

from .helpers import get_logger

logger = get_logger()


class InternalFiles:
    def __init__(self, llama_api_key: str, vendor_api_key: str, max_retries: int = 3):
        self.multimodalparser = LlamaParse(
            api_key=llama_api_key,
            result_type="markdown",
            verbose=True,
            language="en",
            spreadsheet_extract_sub_tables=True,
            output_tables_as_HTML=True,
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt-4-1-mini",
            vendor_multimodal_api_key=vendor_api_key,
        )

        self.parser = LlamaParse(
            api_key=llama_api_key,
            result_type="markdown",
            verbose=True,
            language="en",
            spreadsheet_extract_sub_tables=True,
            output_tables_as_HTML=True,
        )
        self.max_retries = max_retries
        self.failed_files = []

    async def download_file_as_bytes(self, url: str):
        delay = 1
        for attempt in range(1, self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    return resp.content
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"Failed to download {url} after {self.max_retries} attempts. Error: {e}")
                    return None
                print(f"Download failed (attempt {attempt}), retrying in {delay}s... Error: {e}")
                await asyncio.sleep(delay)
                delay *= 2

    async def parse_file(self, file_bytes: bytes, filename: str):
        if filename.endswith(".xlsx") or filename.endswith(".csv"):
            return await self.parser.aparse(file_bytes, extra_info={"file_name": filename})
        else:
            return await self.multimodalparser.aparse(file_bytes, extra_info={"file_name": filename})

    async def process_url(self, url: str, filename: str):
        file_bytes = await self.download_file_as_bytes(url)
        if not file_bytes:
            self.failed_files.append(url)
            return None, url, filename

        try:
            parsed_docs = await self.parse_file(file_bytes, filename)
            return parsed_docs, url, filename
        except Exception as e:
            print(f"Failed parsing {url}: {e}")
            self.failed_files.append(url)
            return None, url, filename

    async def parse_files(self, urls: list[str]):
        url_filename_pairs = []
        for i, file_dict in enumerate(urls):
            fname = file_dict["filename"]
            furl = file_dict["url"]
            url_filename_pairs.append((furl, fname))

        results = await asyncio.gather(*(self.process_url(url, fname) for url, fname in url_filename_pairs))

        langchain_docs = []
        for parsed_docs, url, filename in results:
            if not parsed_docs:
                continue

            json_doc = await parsed_docs.aget_json()
            for i, page in enumerate(json_doc["pages"]):
                try:
                    page_content = convert_to_markdown(page["md"]).replace("\\", "")
                except Exception as e:
                    print(f"convert_to_markdown failed for {filename}, page {i + 1}: {e}")
                    page_content = page["md"].replace("\\", "")

                langchain_docs.append(
                    Document(
                        page_content=page_content,
                        metadata={
                            "url": url,
                            "page": f"page_{i + 1}",
                            "document_type": filename,
                        },
                    )
                )
        return langchain_docs


async def process_internal_files(urls):

    internal = InternalFiles(
        llama_api_key=Secrets.get(SecretKeys.LLAMA_CLOUD_API_KEY),
        vendor_api_key=Secrets.get(SecretKeys.OPENAI_API_KEY),
    )

    docs = await internal.parse_files(urls)

    return docs
