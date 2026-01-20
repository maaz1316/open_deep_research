import asyncio
from datetime import datetime
from io import BytesIO
import re

import dart_fss
import fitz
import httpx
from langchain_core.documents import Document
import requests

from .helpers import get_logger

logger = get_logger()


class DART:
    """
    Asynchronous variant of your DART helper.
    - Keeps execution logic and return shapes identical to your original.
    - Adds async I/O, concurrency limits, retries, and defensive guards.
    """

    def __init__(self, corp_code, api_key, *, max_concurrency=6, timeout=20):
        self.corp_code = corp_code
        self.url_for_dcm = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcp_no}"
        self.pdf_url = "https://dart.fss.or.kr/pdf/download/pdf.do?rcp_no={rcp_no}&dcm_no={dcm_no}"
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) " "AppleWebKit/537.36 (KHTML, like Gecko) " "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        self.list_items = ["I.", "II.", "III.", "IV.", "V.", "VI.", "VII.", "VIII.", "IX.", "X.", "XI.", "XII.", "XIII.", "XIV."]

        self._timeout = timeout
        self._retries = 3
        self._backoff = 0.7
        self._retry_statuses = {429, 500, 502, 503, 504}
        self._sem = asyncio.Semaphore(max_concurrency)

        self._sync_session = requests.Session()
        self._sync_session.headers.update(self.headers)

        self._async_client = None
        if httpx is not None:
            self._async_client = httpx.AsyncClient(headers=self.headers, timeout=self._timeout)

        if dart_fss is not None:
            try:
                dart_fss.set_api_key(api_key=api_key)
            except Exception:
                pass

        self._re_dcm = re.compile(r"dcmNo['\"]?\]\s*=\s*['\"](\d+)['\"]")

    async def aclose(self):
        if self._async_client is not None:
            try:
                await self._async_client.aclose()
            except Exception:
                pass

    async def _aget_text(self, url: str):
        async with self._sem:
            for attempt in range(1, self._retries + 1):
                try:
                    if self._async_client is not None:
                        r = await self._async_client.get(url)
                        if r.status_code == 200 and r.text:
                            return r.text
                        if r.status_code not in self._retry_statuses:
                            return None
                    else:
                        # Fallback: run blocking GET in a thread
                        def _do():
                            return self._sync_session.get(url, timeout=self._timeout)

                        r = await asyncio.to_thread(_do)
                        if r.status_code == 200 and r.text:
                            return r.text
                        if r.status_code not in self._retry_statuses:
                            return None
                except Exception:
                    pass
                await asyncio.sleep(self._backoff * attempt)
        return None

    async def _aget_bytes(self, url: str):
        async with self._sem:
            for attempt in range(1, self._retries + 1):
                try:
                    if self._async_client is not None:
                        r = await self._async_client.get(url)
                        if r.status_code == 200 and r.content:
                            return r.content
                        if r.status_code not in self._retry_statuses:
                            return None
                    else:

                        def _do():
                            return self._sync_session.get(url, timeout=self._timeout)

                        r = await asyncio.to_thread(_do)
                        if r.status_code == 200 and r.content:
                            return r.content
                        if r.status_code not in self._retry_statuses:
                            return None
                except Exception:
                    pass
                await asyncio.sleep(self._backoff * attempt)
        return None

    def get_dates(self):

        try:
            thisyear = datetime.today().strftime("%Y")
            previousyear = str(int(thisyear) - 1)

            start_date_10k = previousyear + "0101"
            end_date_10k = thisyear + "1231"

            start_date_10Q = thisyear + "0101"
            end_date_10Q = datetime.today().strftime("%Y%m%d")

            return start_date_10k, end_date_10k, start_date_10Q, end_date_10Q
        except Exception:
            return "20000101", "20001231", "20000101", "20000101"

    async def get_rcp_no(self, start_date, report_type):

        if dart_fss is None:
            return []
        try:

            def _search():
                return dart_fss.filings.search(corp_code=self.corp_code, bgn_de=start_date, pblntf_detail_ty=report_type)

            reports = await asyncio.to_thread(_search)
        except Exception:
            return []

        try:
            data = reports.to_dict()
            report_list = (data or {}).get("report_list", []) or []
            rcp_no = []
            for report in report_list:
                try:
                    if start_date[:4] in report.get("report_nm", ""):
                        if report.get("rcp_no"):
                            rcp_no.append(report["rcp_no"])
                except Exception:
                    continue
            return rcp_no
        except Exception:
            return []

    async def get_dcm_data(self, rcp_nos):
        if not rcp_nos:
            return []

        async def fetch_one(rcp):
            try:
                url = self.url_for_dcm.format(rcp_no=rcp)
                return await self._aget_text(url)
            except Exception:
                return None

        return await asyncio.gather(*(fetch_one(rcp) for rcp in rcp_nos))

    def get_dcm_no(self, dcm_data):
        dcm_nos = []
        if not dcm_data:
            return dcm_nos
        for i in dcm_data:
            try:
                if not i:
                    dcm_nos.append(None)
                    continue
                m = self._re_dcm.search(i)
                dcm_nos.append(m.group(1) if m else None)
            except Exception:
                dcm_nos.append(None)
        return dcm_nos

    async def get_data(self, rcp_no, dcm_no, document_type, type):

        if document_type in ["a001", "a003"]:
            doc = None
            try:
                pdf_bytes = await self._aget_bytes(self.pdf_url.format(rcp_no=rcp_no, dcm_no=dcm_no))
                if not pdf_bytes:
                    return {} if type == "sections" else []
                pdf_file = BytesIO(pdf_bytes)
                doc = await asyncio.to_thread(fitz.open, stream=pdf_file, filetype="pdf")
            except Exception:
                if doc:
                    try:
                        await asyncio.to_thread(doc.close)
                    except Exception:
                        pass
                return {} if type == "sections" else []

            try:
                try:
                    toc = await asyncio.to_thread(doc.get_toc)
                except Exception:
                    toc = []

                results = {}

                for entry in toc:
                    try:
                        _, title, page = entry if len(entry) >= 3 else (None, None, None)
                        if title is None or page is None:
                            continue
                        match = next((item for item in self.list_items if str(title).startswith(item)), None)
                        if match:
                            key = str(title).replace(match, "", 1).strip()
                            if key:
                                results[key] = {"match": match, "start_page": int(page)}
                    except Exception:
                        continue

                ordered_keys = [k for k in results]

                if not ordered_keys:

                    return {} if type == "sections" else []

                for i in range(len(ordered_keys) - 1):
                    current_key = ordered_keys[i]
                    next_key = ordered_keys[i + 1]
                    try:
                        results[current_key]["end_page"] = max(results[current_key]["start_page"], int(results[next_key]["start_page"]) - 1)

                        if results[current_key].get("match") == "III.":
                            start = int(results[current_key]["start_page"])
                            end = int(results[current_key]["end_page"])
                            sub_section_toc = []
                            for sub, title, page in toc:
                                try:
                                    if start <= int(page) <= end and sub in [1, 2]:
                                        sub_section_toc.append([sub, title, int(page)])
                                except Exception:
                                    continue
                            for level, title, page in sub_section_toc:
                                try:
                                    if "4." in str(title):
                                        results[current_key]["end_page"] = max(start, page - 1)
                                        break
                                except Exception:
                                    continue
                    except Exception:
                        results[current_key]["end_page"] = results[current_key]["start_page"]

                try:
                    results[ordered_keys[-1]]["end_page"] = int(doc.page_count)
                except Exception:
                    results[ordered_keys[-1]]["end_page"] = results[ordered_keys[-1]]["start_page"]

                if type == "sections":

                    for key in ordered_keys:
                        try:
                            start = max(0, int(results[key]["start_page"]) - 1)
                            end = max(start, int(results[key]["end_page"]) - 1)

                            async def _extract_range(s, e):
                                parts = []
                                for page_num in range(s, e + 1):
                                    try:
                                        page = await asyncio.to_thread(doc.load_page, page_num)
                                        text = await asyncio.to_thread(page.get_text)
                                        parts.append(text or "")
                                    except Exception:
                                        parts.append("")
                                return "\n".join(parts)

                            results[key]["text"] = await _extract_range(start, end)
                        except Exception:
                            results[key]["text"] = ""
                    return results

                if type == "page":
                    text_parts = []
                    for key in ordered_keys:
                        try:
                            start = max(0, int(results[key]["start_page"]) - 1)
                            end = max(start, int(results[key]["end_page"]) - 1)
                            for page_num in range(start, end + 1):
                                try:
                                    page = await asyncio.to_thread(doc.load_page, page_num)
                                    text = await asyncio.to_thread(page.get_text)
                                    text_parts.append(text or "")
                                except Exception:
                                    text_parts.append("")
                        except Exception:
                            continue
                    return text_parts

                return []

            finally:
                try:
                    if doc is not None:
                        await asyncio.to_thread(doc.close)
                except Exception:
                    pass

        else:

            doc = None
            try:
                pdf_bytes = await self._aget_bytes(self.pdf_url.format(rcp_no=rcp_no, dcm_no=dcm_no))
                if not pdf_bytes:
                    return []
                pdf_file = BytesIO(pdf_bytes)
                doc = await asyncio.to_thread(fitz.open, stream=pdf_file, filetype="pdf")
                text_parts = []
                i = 0

                for idx in range(doc.page_count):
                    try:
                        if i == 0:
                            i += 1
                            continue
                        page = await asyncio.to_thread(doc.load_page, idx)
                        text = await asyncio.to_thread(page.get_text)
                        text_parts.append(text or "")
                        i += 1
                    except Exception:
                        text_parts.append("")
                        i += 1
                return text_parts
            except Exception:
                return []
            finally:
                try:
                    if doc is not None:
                        await asyncio.to_thread(doc.close)
                except Exception:
                    pass

    async def get_dart_data(self, type):

        try:
            bgn_de_10k, end_de_10k, bgn_de_10q, end_de_10q = self.get_dates()

            rcp_10k, rcp_10q, rcp_f001 = await asyncio.gather(
                self.get_rcp_no(bgn_de_10k, "a001"),
                self.get_rcp_no(bgn_de_10q, "a003"),
                self.get_rcp_no(bgn_de_10k, "F001"),
            )

            dcm_10k = self.get_dcm_no(await self.get_dcm_data(rcp_10k)) if rcp_10k else []
            dcm_10q = self.get_dcm_no(await self.get_dcm_data(rcp_10q)) if rcp_10q else []
            dcm_f001 = self.get_dcm_no(await self.get_dcm_data(rcp_f001)) if rcp_f001 else []

            docs = {}

            async def _proc_one(kind, idx, rcp, dcm, doc_type):
                if dcm is None:
                    return None
                try:
                    filing_data = await self.get_data(rcp, dcm, doc_type, type)
                    key = f"{kind}-{idx}"
                    link = self.url_for_dcm.format(rcp_no=rcp)
                    return key, {"link": link, "content": filing_data}
                except Exception:
                    return None

            tasks = []

            if rcp_10k:
                for idx, (rcp, dcm) in enumerate(zip(rcp_10k, dcm_10k), start=1):
                    tasks.append(_proc_one("10K", idx, rcp, dcm, "a001"))

            if rcp_10q:
                for idx, (rcp, dcm) in enumerate(zip(rcp_10q, dcm_10q), start=1):

                    async def _wrap(idx=idx, rcp=rcp, dcm=dcm):
                        res = await _proc_one("10Q-Q", idx, rcp, dcm, "a003")
                        if res:
                            return (f"10Q-Q{idx}", res[1])
                        return None

                    tasks.append(_wrap())

            if rcp_f001:
                for idx, (rcp, dcm) in enumerate(zip(rcp_f001, dcm_f001), start=1):
                    tasks.append(_proc_one("F001", idx, rcp, dcm, "f001"))

            if tasks:
                results = await asyncio.gather(*tasks)
                for item in results:
                    if not item:
                        continue
                    key, payload = item
                    docs[key] = payload

            return docs
        except Exception:
            return {}


async def get_dart_chunks(docs):
    documents = []
    for document_type, doc in docs.items():
        for page in doc["content"]:
            metadata = {"document_type": document_type, "url": doc["link"]}
            documents.append(Document(page_content=page, metadata=metadata))

    return documents


async def get_dart_docs(corp_code, api_key):
    logger.info("DART feth start")
    dart = DART(corp_code=corp_code, api_key=api_key)
    docs = await dart.get_dart_data("page")
    logger.info("DART fetch complete")
    chunks = await get_dart_chunks(docs)
    return chunks
