from langchain_core.messages import HumanMessage

from infras.secrets.constants import SecretKeys
from libs.secrets.secrets import Secrets

from .configuration import Configuration, InternalFiles, RAGSource, SearchAPI
from .dart import get_dart_docs
from .deep_researcher import deep_researcher
from .helpers import (
    generate_company_summary,
    logger,
    patch_request_url,
    patch_url_function_progress,
    PatchRequest,
    ResponseStatus,
)
from .internal_files import process_internal_files
from .rag_tool import build_rag_tool
from .sec import get_sec_docs
from .state import AgentState
from .utils import template_to_text
from .websearch import get_docs


async def generate_report_agent(
    id=None,
    company_name=None,
    company_first_name=None,
    industry=None,
    competitors=None,
    ticker=None,
    source=None,
    template=None,
    template_obj=None,
    language=None,
    corp_code=None,
    file_urls=None,
    company_url=None,
    web_search=False,
    capiq_search=False,
    report_type=None,
):

    try:
        config = Configuration()
        config.search_api = SearchAPI.PERPLEXITY
        config.allow_clarification = False

        OPENAI_API_KEY = Secrets.get(SecretKeys.OPENAI_API_KEY)
        SEC_API_KEY = Secrets.get(SecretKeys.SEC_API_KEY)
        DART_API_KEY = Secrets.get(SecretKeys.DART_API_KEY)
        ANTHROPIC_API_KEY = Secrets.get(SecretKeys.ANTHROPIC_API_KEY)
        TAVILY_API_KEY = Secrets.get(SecretKeys.TAVILY_API_KEY)
        PERPLEXITY_API_KEY = Secrets.get(SecretKeys.PERPLEXITY_API_KEY)
        GOOGLE_API_KEY = Secrets.get(SecretKeys.GEMINI_API_KEY)

        payload = PatchRequest(id=id, status=ResponseStatus.shortlisting, source=source)
        await patch_url_function_progress(patch_request_url, payload)

        config.rag_source = RAGSource.NONE
        available_tools = []
        chunk_docs = []
        statements = {}
        internal_chunk_docs = []

        patch_answer = PatchRequest(id=id, status=ResponseStatus.downloading_source_file)
        await patch_url_function_progress(patch_request_url, patch_answer)

        if source.lower() == "sec":
            logger.info("Fetching SEC Docs")
            config.rag_source = RAGSource.SEC
            chunk_docs, statements = await get_sec_docs(ticker, SEC_API_KEY)
            if not chunk_docs:
                config.rag_source = RAGSource.WEB
        if source.lower() == "dart":
            logger.info("Fetching DART Docs")
            config.rag_source = RAGSource.DART
            chunk_docs = await get_dart_docs(corp_code, DART_API_KEY)
            if not chunk_docs:
                config.rag_source = RAGSource.WEB
            statements = {}
        if source.lower() == "web" and web_search and not file_urls:
            logger.info("*******************Fetching Docs from WEB*******************")
            logger.info(f"Company URL: {company_url}")
            config.rag_source = RAGSource.WEB
            chunk_docs = await get_docs(company_url)
            statements = {}

        if file_urls:
            logger.info("*******************Processing internal files*******************")
            config.internal_files = InternalFiles.YES
            internal_chunk_docs = await process_internal_files(file_urls)
            statements = {}

        api_keys = {
            "OPENAI_API_KEY": OPENAI_API_KEY,
            "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
            "GOOGLE_API_KEY": GOOGLE_API_KEY,
            "TAVILY_API_KEY": TAVILY_API_KEY,
            "PERPLEXITY_API_KEY": PERPLEXITY_API_KEY,
        }

        if chunk_docs:
            logger.info("***************BUILDING SEC/DART RAG******************")
            rag_tool = build_rag_tool(
                chunk_docs=chunk_docs, openai_api_key=OPENAI_API_KEY, name="company_reports_search", source_label="Company Reports"
            )
            available_tools.append("rag_tool")

        if internal_chunk_docs:
            internal_files_tool = build_rag_tool(
                chunk_docs=internal_chunk_docs,
                name="company_internal_files_search",
                source_label="Company Internal Files",
                batch_size=10,
                openai_api_key=OPENAI_API_KEY,
            )

            config.internal_files_tool = internal_files_tool
            available_tools.append("internal_files")

        if web_search:
            available_tools.append("web_search")

        config_dict = config.model_dump(exclude_none=True, exclude_unset=True)
        config_dict["configurable"] = {
            "recursion_limit": 50,
            "deep_research_agent": False,
            "id": id,
            "apiKeys": api_keys,
        }

        if chunk_docs:
            config_dict["configurable"]["rag_tool"] = rag_tool

        if internal_chunk_docs:
            config_dict["configurable"]["internal_files_tool"] = internal_files_tool

        config_dict["configurable"]["available_tools"] = available_tools
        config_dict["configurable"]["web_search"] = web_search
        config_dict["configurable"]["report_type"] = report_type

        if capiq_search:
            config_dict["configurable"]["capiq_search"] = True
            config_dict["configurable"]["company_name"] = company_name

        logger.info(f"******AVAILABLE TOOLS******\n\n{' '.join(available_tools)}")

        company_summary = await generate_company_summary(company_url)

        brief = """I want to generate Information Memorandum (IM) for a company.
        Company Name: {company_name}
        Company Info: {company_summary}
        Company Industry: {industry}
        Company URL: {company_url}
        Competitors: {competitors}


        Here is the guide for IM sections and description for each section:
        {sections}
        """.format(
            sections=template_to_text(template_obj),
            company_name=company_name,
            industry=industry,
            company_url=company_url,
            competitors=competitors,
            company_summary=company_summary,
        )

        deep_researcher_test_state = AgentState(
            messages=[HumanMessage(content=brief)],
            company_name=company_name,
            industry=", ".join(industry),
            competitors=", ".join(competitors),
            sections=template,
            company_summary=company_summary,
            income_statement=statements.get("income_statement", ""),
            balance_sheet=statements.get("balance_sheet", ""),
            cashflow_statement=statements.get("cashflow_statement", ""),
            language=language,
        )

        result = await deep_researcher.ainvoke(deep_researcher_test_state, config=config_dict)
        logger.info("*******RESULT*******\n\n", result)
    except Exception as e:
        logger.error(f"Failed to generate report: {str(e)}", exc_info=True)
        payload = PatchRequest(id=id, status=ResponseStatus.exception)
        await patch_url_function_progress(patch_request_url, payload)
        raise e
