from typing import List

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from infras.secrets.constants import SecretKeys
from libs.secrets.secrets import Secrets

from .prompts import transform_messages_into_research_topic_prompt
from .utils import get_today_str


class Sections(BaseModel):
    section_name: str = Field(description="name of the section")
    section_details: str = Field(description="details of the section including brief and sub-sections")


class IMSections(BaseModel):
    brief: str = Field(description="brief of research plan")
    sections: List[Sections]


async def write_research_brief(input):

    query_llm = ChatOpenAI(
        model="gpt-4.1",
        temperature=0,
        api_key=Secrets.get(SecretKeys.OPENAI_API_KEY),
    )

    research_model = query_llm.with_structured_output(IMSections)

    result = await research_model.ainvoke(transform_messages_into_research_topic_prompt.format(messages=input, date=get_today_str()))

    return result
