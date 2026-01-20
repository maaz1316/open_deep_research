# pylint: disable=all
# fmt: off
# flake8: noqa
from enum import IntEnum
from typing import Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel


class ResponseStatus(IntEnum):
    start=1
    industry_and_competitors=2
    shortlisting= 3
    downloading_source_file=4
    logs_start=5
    logs_stream=6
    report_start=7
    report_stream=8
    references=9
    complete=10
    exception=11
    summary=12



class WorkflowInput(BaseModel):
    id: str
    url: Optional[str] = ""
    company_url: Optional[str]= ""
    source: Optional[str] = ""
    template: Optional[List[str]] = []
    fullname: Optional[str]= ""
    ticker: Optional[str]= ""
    industry: Optional[List[str]]= ""
    competitors :Optional[List[str]]=[]
    first_name: Optional[str]= ""
    language: Optional[str]=""
    corp_code: Optional[str] =""
    file_urls: Optional[List[str]] = []
    file_details: Optional[List[Dict]] = {}
    web_search: Optional[bool] = False
    capiq_search: Optional[bool] = False




class PatchRequest(BaseModel):
    id: str
    status: int
    industry: Optional[str]= ""
    competitors: Optional[List[str]] = []
    logs: Optional[Dict] = None
    reportContent: Optional[str] = None
    summaryContent: Optional[str] = None
    references: Optional[List[str]] = []
    fullname: Optional[str] = ""
    ticker: Optional[str] = ""
    first_name: Optional[str] = ""
    source: Optional[str] = ""
    language: Optional[str] =""
    corp_code: Optional[str]=""
