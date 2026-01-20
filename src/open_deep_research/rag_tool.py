import asyncio
from collections import defaultdict

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
from typing_extensions import Dict, List, Optional, Tuple, Union

from infras.secrets.constants import SecretKeys
from libs.secrets.secrets import Secrets

from .helpers import get_logger
from .prompts import enhance_query_prompt, rag_answer_prompt

logger = get_logger()

llm_semaphore = asyncio.Semaphore(10)

fallback_llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=0,
    api_key=Secrets.get(SecretKeys.OPENAI_API_KEY),
)


async def safe_llm_call(model, messages, timeout=60, fallback=None):
    """Run an LLM safely with timeout + semaphore + fallback."""
    async with llm_semaphore:
        try:
            return await asyncio.wait_for(model.ainvoke(messages), timeout=timeout)
        except asyncio.TimeoutError:
            if fallback:
                try:
                    return await asyncio.wait_for(fallback.ainvoke(messages), timeout=timeout)
                except Exception as e:
                    logger.info(f"Fallback LLM failed: {e}")
                    return None
            return None
        except Exception as e:
            logger.info(f"LLM call failed: {e}")
            return None


class HybridRRF:
    def __init__(self, retrievers, k: int = 60, debug: bool = False):
        """
        retrievers: list of retriever objects (vectorstore, BM25, etc.)
        k: RRF parameter (default 60)
        """
        self.retrievers = retrievers
        self.k = k
        self.debug = debug

    async def ainvoke(self, queries: List[str], intent: str, per_type: int = 3) -> List[Document]:
        assert len(queries) == len(self.retrievers), "Queries must match retrievers"

        tasks = []
        for retriever, query in zip(self.retrievers, queries):
            if hasattr(retriever, "ainvoke"):
                tasks.append(retriever.ainvoke(query))
            else:
                tasks.append(asyncio.to_thread(retriever.invoke, query))

        ranked_lists = await asyncio.gather(*tasks, return_exceptions=True)
        ranked_lists = [rl for rl in ranked_lists if isinstance(rl, list)]

        weights = self._intent_weights(intent)

        scored_docs = self._rrf(ranked_lists, weights)
        top_n = self._top_n_per_type(scored_docs, per_type)
        return top_n

    def _intent_weights(self, intent: str) -> List[float]:
        mapping = {
            "executives": [0.7, 0.3],
            "financials": [0.5, 0.5],
            "risks": [0.5, 0.5],
            "business": [0.6, 0.4],
            "governance": [0.6, 0.4],
            "legal": [0.5, 0.5],
            "operations": [0.5, 0.5],
            "other": [0.5, 0.5],
        }
        return mapping.get((intent or "other").lower(), [0.5, 0.5])

    def _rrf(self, lists: List[List[Document]], weights: List[float]) -> List[Tuple[Document, float]]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        for retr_idx, docs in enumerate(lists):
            for rank, doc in enumerate(docs, start=1):
                key = doc.page_content + str(doc.metadata.get("document_type", ""))
                doc_map[key] = doc
                scores[key] = scores.get(key, 0) + weights[retr_idx] * (1.0 / (self.k + rank))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_map[key], score) for key, score in ranked]

    def _top_n_per_type(self, scored_docs: List[Tuple[Document, float]], per_type: int) -> List[Document]:
        grouped: Dict[str, List[Tuple[Document, float]]] = defaultdict(list)
        final: List[Document] = []

        for doc, score in scored_docs:
            dtype = doc.metadata.get("document_type", "unknown")
            grouped[dtype].append((doc, score))

        for dtype, docs in grouped.items():
            docs_sorted = sorted(docs, key=lambda x: x[1], reverse=True)[:per_type]
            for doc, score in docs_sorted:
                doc.metadata["rrf_score"] = score
                final.append(doc)

        return final


class EnhancedQuery(BaseModel):
    enhanced_query: str = Field(None, description="Improved natural language query reformulated")
    keywords: List[str] = Field(
        None,
        description=(
            "Concise list of 8–12 critical keywords or phrases derived from the user query,"
            " optimized for lexical/sparse retrieval (e.g., BM25)"
        ),
    )
    intent: Optional[str] = Field(
        None,
        description=(
            "High-level intent category of the query (e.g., financials," " executives, risks, business, legal, governance, operations)"
        ),
    )


async def enhance_query(input, query_llm):
    query_llm_structured = query_llm.with_structured_output(EnhancedQuery)

    prompt = enhance_query_prompt.format(input=input)

    resp = await safe_llm_call(query_llm_structured, [HumanMessage(content=prompt)], timeout=30, fallback=fallback_llm)

    if resp is None:
        logger.info("Enhance Query failed")
        return input

    return resp


def build_rag_tool(
    chunk_docs,
    name: str = "rag_search",
    source_label: str = "",
    batch_size: int = 150,
    dense_k: int = 100,
    dense_fetch_k: int = 150,
    bm25_k: int = 100,
    embedding_model: str = "text-embedding-3-large",
    llm_model: str = "gpt-4.1-mini",
    query_model: str = "gpt-4.1-mini",
    openai_api_key: Optional[str] = None,
):
    """
    Build a RAG tool for a company's annual filings.
    Returns:
        rag_tool: LangChain @tool ready for use
    """
    logger.info("Building RAG")
    embedding = OpenAIEmbeddings(model=embedding_model, api_key=openai_api_key)
    vectorstore = Chroma(embedding_function=embedding)

    for i in range(0, len(chunk_docs), batch_size):
        vectorstore.add_documents(chunk_docs[i : i + batch_size])

    dense_retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": dense_k, "fetch_k": dense_fetch_k})
    sparse_retriever = BM25Retriever.from_documents(chunk_docs, k=bm25_k)
    ensemble = HybridRRF([dense_retriever, sparse_retriever], k=60, debug=False)

    llm = ChatOpenAI(model=llm_model, temperature=0, api_key=openai_api_key)
    query_llm = ChatOpenAI(model=query_model, temperature=0, api_key=openai_api_key)
    fallback_llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0, api_key=openai_api_key)

    sys = SystemMessage(content=(rag_answer_prompt))

    async def synthesize(query: str, docs: List[Document], source_label) -> str:

        class SourceAnswers(BaseModel):
            content: str = Field(description="Answer to the question if the content is present")
            url: str = Field(description="url of the source")
            title: str = Field(description="title of the document verbatim.")

        class Answers(BaseModel):
            document_type: str = Field(description="Type of the document")
            answers: List[SourceAnswers]

        if not docs:
            return "No information retrieved."
        # context = "\n**********************************\n\n".join(d.page_content for d in docs)
        grouped_docs = defaultdict(list)

        for d in docs:
            url = d.metadata.get("url", "Not available")
            title = d.metadata.get("document_type", "Not available")
            grouped_docs[(url, title)].append(d.page_content)

        context = "\n\n***********************************\n\n".join(
            "\n".join(chunks) + f"\n\nTitle: {title}\nURL: {url}" for (url, title), chunks in grouped_docs.items()
        )

        msg = HumanMessage(content=f"Question:\n{query}\n\\Document:\n{context}\n\nDocument Type:{source_label}")
        resp = await safe_llm_call(
            llm.with_structured_output(Answers), [sys, msg], timeout=60, fallback=fallback_llm.with_structured_output(Answers)
        )
        return resp.model_dump_json() if resp else "No answer available"

    # --- The unified tool ---
    @tool(parse_docstring=True)
    async def rag_tool(
        query: Union[str, List[str]],
        k: int = 4,
    ) -> str:
        """
        Fetch and summarize information from company's documents.
        Can process one query or multiple queries in parallel.

        Args:
            query: Natural language question OR list of questions.
            k: Number of top docs to retrieve for each query.

        Returns:
            Combined answers + citations.
        """
        queries = [query] if isinstance(query, str) else query
        results = []

        async def handle_one(q: str) -> str:
            logger.info(f"RAG Called {source_label}:{q}")
            enriched_query = await enhance_query(q, query_llm)
            docs: List[Document] = await ensemble.ainvoke(
                [enriched_query.enhanced_query, " ".join(enriched_query.keywords)],
                enriched_query.intent,
                k,
            )
            answer = await synthesize(enriched_query.enhanced_query, docs, source_label)

            # seen = set()
            # for d in docs:
            #     meta = d.metadata or {}
            #     src = meta.get("url") or meta.get("source")
            #     if src:
            #         seen.add(src)

            # formatted_output = f"{source_label} results: \n\n"
            # formatted_output += answer
            # formatted_output += "URLs:\n" + "\n".join(seen) + "\n\n"

            # return formatted_output

            return answer

        results = await asyncio.gather(*[handle_one(q) for q in queries], return_exceptions=True)
        logger.info(f"RAG RESULTS: {"\n\n---\n\n".join(r for r in results)}")
        return "\n\n---\n\n".join(r for r in results)

    rag_tool.name = name
    return rag_tool
