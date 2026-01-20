# pylint: disable=all
# fmt: off
# flake8: noqa
import asyncio
import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

# from apps.domains.im_workflow.dart_agent import (
#     DartCompanyNotFoundException,
#     DartDownloadException,
# )
from apps.domains.im_workflow.patch_requests import patch_url_function_progress

# from apps.domains.llmchat.parallel_langgraph_agent import process_user_request
# from apps.domains.llmchat.schemas import llmchat_input, llmchat_output
from apps.domains.im_workflow.prom_functions import generate_company_information

# from apps.domains.im_workflow.prom import generate_company_information,
from apps.domains.im_workflow.schemas import ResponseStatus, WorkflowInput
from apps.domains.im_workflow.util_functions import get_logger
from apps.domains.open_deep_research_main.src.open_deep_research import (
    generate_report_agent,
)
from apps.domains.open_deep_research_main.src.open_deep_research.helpers import (
    patch_url_function_progress,
    PatchRequest,
)
from infras.secrets.constants import SecretKeys
from libs.secrets.secrets import Secrets

# from apps.domains.im_workflow.schemas import PatchRequest

# Configure logger
logger = get_logger()

# Load environment variables from .env file
BASE_URL=Secrets.get(SecretKeys.BASE_URL)
url = f"""{BASE_URL}/api/report/im-workflow"""
# url = f"""{BASE_URL}/api/report/im-workflow"""

@csrf_exempt
@require_http_methods(["POST"])
async def process_im_request(request):
    print("**************REQUEST RECEIVED**************************")
    workflow_request = None
    try:
        request_data = json.loads(request.body.decode("utf-8"))
        logger.info(request_data)
        workflow_request = WorkflowInput(**request_data)
        if workflow_request.template == []:
            if workflow_request.source in ["SEC","DART","WEB"]:
                asyncio.create_task( generate_company_information(workflow_request.id, workflow_request.url, workflow_request.source,workflow_request.language))
        else:
            if workflow_request.source=="SEC":
                asyncio.create_task(generate_report_agent(company_url = workflow_request.company_url, id = workflow_request.id, company_name=workflow_request.fullname, industry = workflow_request.industry, competitors=workflow_request.competitors, ticker=workflow_request.ticker, source=workflow_request.source,template=workflow_request.template, language = workflow_request.language, file_urls=workflow_request.file_details, web_search=workflow_request.web_search, capiq_search=workflow_request.capiq_search ))
            elif workflow_request.source=="DART":
                asyncio.create_task(generate_report_agent(company_url = workflow_request.company_url, id = workflow_request.id, company_name=workflow_request.fullname, company_first_name = workflow_request.first_name, industry = workflow_request.industry, competitors = workflow_request.competitors,  source = workflow_request.source, template = workflow_request.template, language = workflow_request.language,corp_code=workflow_request.corp_code, file_urls=workflow_request.file_details, web_search=workflow_request.web_search, capiq_search=workflow_request.capiq_search))
            if workflow_request.source=="WEB":
                asyncio.create_task(generate_report_agent(company_url = workflow_request.company_url, id = workflow_request.id, company_name=workflow_request.fullname, industry = workflow_request.industry, competitors=workflow_request.competitors, source=workflow_request.source,template=workflow_request.template, language = workflow_request.language, file_urls=workflow_request.file_details, web_search=workflow_request.web_search, capiq_search=workflow_request.capiq_search))

        return JsonResponse(
            {
                "success": True,
                "message": "LLM Message Fetch Successful",
            },
            status=200,
        )

    except Exception as e:
        request_data = json.loads(request.body.decode("utf-8"))
        workflow_request = WorkflowInput(**request_data)
        logger.exception(f"An unexpected error occurred during chat request processing.: {e}")
        payload = PatchRequest(id=workflow_request.id, status=ResponseStatus.exception)
        asyncio.create_task(patch_url_function_progress(url, payload))
        raise e
        # return JsonResponse({"success": False, "message": f"An internal server error occurred {e}"}, status=500)
