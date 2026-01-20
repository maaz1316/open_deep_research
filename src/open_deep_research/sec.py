import asyncio
from datetime import datetime
import json
import random
import re

from html_to_markdown import convert_to_markdown
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import requests
from sec_api import ExtractorApi, QueryApi, XbrlApi
from typing_extensions import Any, Dict, List, Tuple

from .helpers import get_logger

logger = get_logger()


class SEC:

    def __init__(
        self,
        ticker: str,
        api_key: str,
        *,
        max_concurrency: int = 6,
        retries: int = 3,
        backoff: float = 0.7,
        timeout: float | None = None,
    ):
        if not isinstance(ticker, str) or not ticker.strip():
            raise ValueError("ticker must be a non-empty string.")
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string.")

        self.item_ids_10K = {
            "1": "Business",
            "1A": "\tRisk Factors",
            "1B": "Unresolved Staff Comments",
            "1C": "Cybersecurity",
            "2": "Properties",
            "3": "Legal Proceedings",
            "4": "Mine Safety Disclosures",
            "5": "Market For Registrant's Common Equity, Related Stockholder Matters, and Issuer Purchases of Equity Securities",
            "6": "[Reserved]",
            "7": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
            "7A": "Quantitative and Qualitative Disclosures About Market Risk",
            "8": "Financial Statements and Supplementary Data",
            "9": "Changes in and Disagreements with Accountants on Accounting and Financial Disclosure",
            "9A": "Controls and Procedures",
            "9B": "Other Information",
            "10": "Directors, Executive Officers and Corporate Governance",
            "11": "Executive Compensation",
            "12": "Security Ownership of Certain Beneficial Owners and Management and Related Stockholder Matters",
            "13": "Certain Relationships and Related Transactions, and Director Independence",
            "14": "Principal Accountant Fees and Services",
            "15": "Exhibit and Financial Statement Schedules",
        }

        self.item_ids_10Q = {
            "part1item1": "Financial Statements (Unaudited) + Notes to consolidated Financial Statements",
            "part1item2": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
            "part1item3": "Quantitative and Qualitative Disclosures About Market Risk",
            "part1item4": "Controls and Procedures",
            "part2item1": "Legal Proceedings",
            "part2item1a": "Risk Factors",
            "part2item2": "Unregistered Sales of Equity Securities and Use of Proceeds",
            "part2item3": "Defaults Upon Senior Securities",
            "part2item4": "Mine Safety Disclosures",
            "part2item5": "Other Information",
            "part2item6": "Exhibits",
        }

        self.ticker = ticker.upper().strip()
        self.queryapi = QueryApi(api_key=api_key)
        self.extractorapi = ExtractorApi(api_key=api_key)
        self.xbrlApi = XbrlApi(api_key=api_key)

        self._sem = asyncio.Semaphore(max_concurrency)
        self._retries = max(1, int(retries))
        self._backoff = float(backoff)
        self._timeout = timeout
        self._retry_statuses = {429, 500, 502, 503, 504}

    async def _call_threaded(self, func, *args, **kwargs):
        if self._timeout:
            return await asyncio.wait_for(asyncio.to_thread(func, *args, **kwargs), timeout=self._timeout)
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _retry_call(self, label: str, func, *args, **kwargs):
        for attempt in range(1, self._retries + 1):
            try:
                async with self._sem:
                    return await self._call_threaded(func, *args, **kwargs)
            except asyncio.CancelledError:
                raise
            except requests.exceptions.HTTPError as http_err:  # type: ignore[attr-defined]
                status = getattr(getattr(http_err, "response", None), "status_code", None)
                if status in self._retry_statuses:
                    return None
            await asyncio.sleep(self._backoff * attempt + random.random() * 0.2)

        return None

    @staticmethod
    def _safe_dt_label(date_str: str, kind: str) -> str:
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            return date_str
        if kind == "year":
            return f"FY{dt.year}"
        q = (dt.month - 1) // 3 + 1
        return f"Q{q}{dt.year}"

    @staticmethod
    def _safe_dt_label_cf(date_str: str, kind: str) -> str:
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            return date_str
        if kind == "year":
            return f"FY{dt.year}"
        q = (dt.month - 1) // 3 + 1
        return f"Q{q}-{dt.year}"

    @staticmethod
    def _dropna_rows(df: pd.DataFrame, frac_keep: float = 0.7) -> pd.DataFrame:
        if df.empty:
            return df
        thresh = max(1, int(frac_keep * df.shape[1]))
        return df.dropna(axis=0, thresh=thresh)

    @staticmethod
    def _to_markdown_or_csv(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(floatfmt=".0f")
        except Exception:
            return df.to_csv(float_format="%.0f")

    @staticmethod
    def _safe_get(d: Dict[str, Any], key: str, default):
        val = d.get(key, default)
        return default if val is None else val

    def get_income_statement(self, xbrl_json: Dict[str, Any], type: str) -> pd.DataFrame:
        store: Dict[str, pd.Series] = {}
        section = self._safe_get(xbrl_json, "StatementsOfIncome", {})
        if not isinstance(section, dict):
            section = {}
        for usGaapItem, facts in section.items():
            vals, idxs = [], []
            if not isinstance(facts, list):
                continue
            for fact in facts:
                if not isinstance(fact, dict) or "segment" in fact:
                    continue
                period = fact.get("period", {})
                index = period.get("endDate") or period.get("instant")
                if not index or index in idxs:
                    continue
                vals.append(fact.get("value", 0))
                idxs.append(index)
            if idxs:
                store[usGaapItem] = pd.Series(vals, index=idxs)

        df = pd.DataFrame(store)
        df = self._dropna_rows(df).T
        if type == "year":
            df = df.rename(columns=lambda c: self._safe_dt_label(c, "year"))
        else:
            df = df.rename(columns=lambda c: self._safe_dt_label(c, "quarter"))
        return df

    def get_balance_sheet(self, xbrl_json: Dict[str, Any], type: str) -> pd.DataFrame:
        store: Dict[str, pd.Series] = {}
        section = self._safe_get(xbrl_json, "BalanceSheets", {})
        if not isinstance(section, dict):
            section = {}
        for usGaapItem, facts in section.items():
            vals, idxs = [], []
            if not isinstance(facts, list):
                continue
            for fact in facts:
                if not isinstance(fact, dict) or "segment" in fact:
                    continue
                period = fact.get("period", {})
                index = period.get("instant") or period.get("endDate")
                if not index or index in idxs:
                    continue
                vals.append(fact.get("value", 0))
                idxs.append(index)
                store[usGaapItem] = pd.Series(vals, index=idxs)

        df = pd.DataFrame(store)
        df = self._dropna_rows(df).T
        if type == "year":
            df = df.rename(columns=lambda c: self._safe_dt_label(c, "year"))
        else:
            df = df.rename(columns=lambda c: self._safe_dt_label(c, "quarter"))
        return df

    def get_cashflow_statement(self, xbrl_json: Dict[str, Any], type: str) -> pd.DataFrame:
        store: Dict[str, pd.Series] = {}
        section = self._safe_get(xbrl_json, "StatementsOfCashFlows", {})
        if not isinstance(section, dict):
            section = {}
        for usGaapItem, facts in section.items():
            vals, idxs = [], []
            if not isinstance(facts, list):
                continue
            for fact in facts:
                if not isinstance(fact, dict) or "segment" in fact:
                    continue
                period = fact.get("period", {})
                index = period.get("endDate") or period.get("instant")
                if not index or index in idxs:
                    continue
                vals.append(fact.get("value", 0))
                idxs.append(index)
            if idxs:
                store[usGaapItem] = pd.Series(vals, index=idxs)

        df = pd.DataFrame(store)
        df = self._dropna_rows(df).T
        if type == "year":
            df = df.rename(columns=lambda c: self._safe_dt_label_cf(c, "year"))
        else:
            df = df.rename(columns=lambda c: self._safe_dt_label_cf(c, "quarter"))
        return df

    def get_dates(self) -> Tuple[str, str, str, str]:
        thisyear = datetime.today().strftime("%Y")
        previousyear = str(int(thisyear) - 1)
        return (
            f"{previousyear}-01-01",
            f"{thisyear}-12-31",
            f"{thisyear}-01-01",
            datetime.today().strftime("%Y-%m-%d"),
        )

    async def aget_filing_urls(self, form_type: str, start_date: str, end_date: str) -> List[str]:
        search_query = f'ticker:{self.ticker} AND formType:"{form_type}" AND filedAt:[{start_date} TO {end_date}]'
        parameters: Dict[str, Any] = {
            "query": search_query,
            "from": "0",
            "size": "10",
            "sort": [{"filedAt": {"order": "desc"}}],
        }
        response = await self._retry_call("get_filings", self.queryapi.get_filings, parameters) or {}
        try:
            filings = response.get("filings", []) or []
            urls: List[str] = []
            year_hint = start_date.split("-")[0]
            for f in filings:
                link = f.get("linkToFilingDetails")
                if link and year_hint in link:
                    urls.append(link)
            return urls
        except Exception:
            return []

    async def aget_filings_data(self) -> Tuple[Dict[str, Any], Dict[str, str]]:
        start_10k, end_10k, start_10q, end_10q = self.get_dates()

        q_10, k_10 = await asyncio.gather(
            self.aget_filing_urls("10-Q", start_10q, end_10q),
            self.aget_filing_urls("10-K", start_10k, end_10k),
        )

        docs: Dict[str, Any] = {}
        cashflow_statements: List[pd.DataFrame] = []
        balancesheets: List[pd.DataFrame] = []
        income_statements: List[pd.DataFrame] = []

        def _doc_ok(section_map: Dict[str, str]) -> bool:
            return bool(section_map) and any(isinstance(v, str) and v.strip() for v in section_map.values())

        async def _concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
            frames = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
            if not frames:
                return pd.DataFrame()
            try:
                return pd.concat(frames, axis=1, sort=False)
            except Exception:
                try:
                    common = set(frames[0].columns)
                    for f in frames[1:]:
                        common &= set(f.columns)
                    if common:
                        frames = [f[list(common)] for f in frames]
                    return pd.concat(frames, axis=1, sort=False)
                except Exception:
                    return pd.DataFrame()

        if k_10:
            k_url = k_10[0]

            async def _fetch_section(key, title):
                html = await self._retry_call("get_section(10-K)", self.extractorapi.get_section, k_url, key, "html")
                return title, (html if isinstance(html, str) else "")

            sec_tasks = [asyncio.create_task(_fetch_section(k, v)) for k, v in self.item_ids_10K.items()]
            local_sections: Dict[str, str] = {}
            for t in sec_tasks:
                try:
                    title, html = await t
                    local_sections[title] = html
                except Exception:
                    pass

            if _doc_ok(local_sections):
                docs["10K"] = {"link": k_url, "content": dict(local_sections)}

            xjson = await self._retry_call("xbrl_to_json(10-K)", self.xbrlApi.xbrl_to_json, k_url) or {}
            try:
                cf = self.get_cashflow_statement(xjson, "year")
                bs = self.get_balance_sheet(xjson, "year")
                is_ = self.get_income_statement(xjson, "year")
                if not cf.empty:
                    cashflow_statements.append(cf)
                if not bs.empty:
                    balancesheets.append(bs)
                if not is_.empty:
                    income_statements.append(is_)
            except Exception as e:
                logger.info(f"Building 10-K statements failed: {e}")

        if q_10:
            for idx, q in enumerate(reversed(q_10), start=1):

                async def _fetch_q_section(key, title):
                    html = await self._retry_call("get_section(10-Q)", self.extractorapi.get_section, q, key, "html")
                    return title, (html if isinstance(html, str) else "")

                sec_tasks = [asyncio.create_task(_fetch_q_section(k, v)) for k, v in self.item_ids_10Q.items()]
                local_sections: Dict[str, str] = {}
                for t in sec_tasks:
                    try:
                        title, html = await t
                        local_sections[title] = html
                    except Exception:
                        pass

                if _doc_ok(local_sections):
                    docs[f"10Q-Q{idx}"] = {"link": q, "content": dict(local_sections)}

                xjson = await self._retry_call("xbrl_to_json(10-Q)", self.xbrlApi.xbrl_to_json, q) or {}
                try:
                    cf = self.get_cashflow_statement(xjson, "quarter")
                    bs = self.get_balance_sheet(xjson, "quarter")
                    is_ = self.get_income_statement(xjson, "quarter")
                    if not cf.empty:
                        cashflow_statements.append(cf)
                    if not bs.empty:
                        balancesheets.append(bs)
                    if not is_.empty:
                        income_statements.append(is_)
                except Exception as e:
                    logger.info(f"Building 10-Q statements failed: {e}")

        docs_result: Dict[str, Any] = docs if docs else {}

        statements_result: Dict[str, str] = {}
        cashflow_statement = await _concat_or_empty(cashflow_statements)
        balancesheet = await _concat_or_empty(balancesheets)
        income_statement = await _concat_or_empty(income_statements)

        if not income_statement.empty:
            statements_result["income_statement"] = self._to_markdown_or_csv(income_statement)
        if not cashflow_statement.empty:
            statements_result["cashflow_statement"] = self._to_markdown_or_csv(cashflow_statement)
        if not balancesheet.empty:
            statements_result["balance_sheet"] = self._to_markdown_or_csv(balancesheet)

        return docs_result, (statements_result if statements_result else {})


class Chunks:

    def __init__(self, docs):
        self.docs = docs
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=200, chunk_overlap=50, encoding_name="cl100k_base"
        )

    def split_html_pages(self, html: str):
        """
        Split an exported-HTML document into pages.
        Supports CSS page-breaks (<hr ... page-break-*> and break-*) and
        common PDF->HTML page container divs. Falls back to 1 page.
        """
        if not isinstance(html, str) or not html.strip():
            return []

        html = html.strip()

        # Regex for <hr> page breaks
        pattern_hr = r"(?is)<hr\b[^>]*" r"\b(?:page-break-(?:after|before)|break-(?:after|before))" r"\s*:\s*(?:always|page)\b[^>]*\/?>\s*"
        hr_sep = re.compile(pattern_hr)

        parts = re.split(hr_sep, html)
        pages = [p.strip() for p in parts if re.search(r"[A-Za-z0-9]", p)]
        if len(pages) > 1:
            return pages

        # Regex lookaheads for page-like divisions
        lookaheads = [
            (
                r"(?is)(?=<(?:div|p|section|article|span|table)\b[^>]*"
                r"\b(?:page-break-(?:after|before)|break-(?:after|before))"
                r"\s*:\s*(?:always|page)\b[^>]*>)"
            ),
            (r"(?is)(?=<(?:div|section)\b[^>]*" r"\bheight\s*:\s*(?:792|841\.89|842)pt\b[^>]*>)"),
            (r"(?is)(?=<(?:div|section)\b[^>]*" r'class="[^"]*\bpage\b[^"]*"[^>]*>)'),
            (r"(?is)(?=<(?:div|section)\b[^>]*" r'id="page-\d+"\b[^>]*>)'),
        ]

        for pat in lookaheads:
            parts = re.split(pat, html)
            pages = [p.strip() for p in parts if re.search(r"[A-Za-z0-9]", p)]
            if len(pages) > 1:
                return pages

        # 3) Fallback
        return [html]

    def split_sections(self, md):
        try:
            if not isinstance(md, str):
                return []
            return [p.strip() for p in re.split(r"(?:^|\n)\s*---\s*(?:\n|$)", md) if p.strip()]
        except Exception:
            return []

    def parse_table(self, md_table):
        rows = []
        try:
            if not isinstance(md_table, str):
                return rows
            for line in md_table.splitlines():
                if "|" not in line:
                    continue
                rows.append([c.strip() for c in line.strip().strip("|").split("|")])
        except Exception as e:
            logger.info(f"parse_table error: {e}")
            # logger.info(f"parse_table error: {e}")
        return rows

    def clean_table(self, matrix):
        try:
            if not matrix:
                return matrix
            # normalize width
            w = max(len(r) for r in matrix if isinstance(r, list))
            mat = []
            for r in matrix:
                r = r if isinstance(r, list) else []
                mat.append([("" if c is None else str(c)) for c in r] + [""] * (w - len(r)))
            # drop all-empty cols
            keep = []
            for j in range(w):
                try:
                    if any(cell.strip("-: ") for cell in (r[j] for r in mat)):
                        keep.append(j)
                except Exception:
                    continue
            mat = [[row[j] for j in keep] for row in mat] if keep else []

            j = 0
            while mat and j < len(mat[0]) - 1:
                try:
                    col, nxt = [r[j] for r in mat], [r[j + 1] for r in mat]
                    is_curr = all((c in {"$", "US$", "USD", ""}) for c in col if c != "")
                    is_num = any(re.fullmatch(r"[()\-\u2212]?\s*[\d,]+(\.\d+)?", (x or "")) for x in nxt)
                    is_pct = all((x == "%" or x == "") for x in nxt)
                    is_num2 = any(re.fullmatch(r"[()\-\u2212]?\s*[\d,]+(\.\d+)?", (x or "")) for x in col)
                    if is_curr and is_num:
                        for r in mat:
                            r[j] = (r[j] + " " + r[j + 1]).strip()
                            r.pop(j + 1)
                    elif is_num2 and is_pct:
                        for r in mat:
                            r[j] = (r[j] + "%").strip()
                            r.pop(j + 1)
                    else:
                        j += 1
                except Exception:
                    j += 1

            def check_any(lst):
                try:
                    s = " ".join(map(str, lst))
                    return bool(re.search(r"[A-Za-z0-9]", s))
                except Exception:
                    return True

            mat_parsed = [i for i in mat if check_any(i)]
            return mat_parsed
        except Exception:
            return []

    def to_md_table(self, matrix):
        try:
            if not matrix:
                return ""
            n = len(matrix[0]) if matrix[0] else 0
            if n <= 0:
                return ""
            out = ["| " + " | ".join(matrix[0]) + " |", "| " + " | ".join(["---"] * n) + " |"]
            for row in matrix[1:]:
                row = list(row) + [""] * (n - len(row)) if len(row) < n else row[:n]
                out.append("| " + " | ".join(row) + " |")
            return "\n".join(out)
        except Exception:
            return ""

    def extract_ordered_chunks(self, md):
        """
        Returns a list like:
        [{"id":"sec0_blk0","type":"narrative","text":...},
        {"id":"sec0_blk1","type":"table","table":..., "notes_before":..., "notes_after":...}, ...]
        """
        chunks = []
        try:
            for si, section in enumerate(self.split_sections(md)):
                try:
                    lines = section.splitlines()
                except Exception:
                    continue
                i, block_id = 0, 0
                while i < len(lines):
                    try:
                        ln = lines[i]
                        if isinstance(ln, str) and ln.strip().startswith("|"):
                            t_start = i
                            t_buf = []
                            while i < len(lines) and isinstance(lines[i], str) and lines[i].strip().startswith("|"):
                                t_buf.append(lines[i])
                                i += 1
                            raw_table = "\n".join(t_buf)

                            nb = []
                            k = t_start - 1
                            while k >= 0 and isinstance(lines[k], str) and lines[k].strip() and not lines[k].strip().startswith("|"):
                                nb.append(lines[k])
                                k -= 1
                            notes_before = "\n".join(reversed(nb)).strip()

                            na = []
                            j = i
                            while (
                                j < len(lines) and isinstance(lines[j], str) and lines[j].strip() and not lines[j].strip().startswith("|")
                            ):
                                na.append(lines[j])
                                j += 1
                            notes_after = "\n".join(na).strip()
                            i = j

                            matrix = self.parse_table(raw_table)
                            matrix = self.clean_table(matrix)
                            cleaned = self.to_md_table(matrix)

                            chunks.append(
                                {
                                    "id": f"sec{si}_blk{block_id}",
                                    "type": "table",
                                    "table": cleaned,
                                    "notes_before": notes_before or None,
                                    "notes_after": notes_after or None,
                                }
                            )
                            block_id += 1

                        else:
                            nb = []
                            while i < len(lines) and not (isinstance(lines[i], str) and lines[i].strip().startswith("|")):
                                nb.append(lines[i])
                                i += 1
                                # stop at hard blank line to keep blocks readable
                                if nb and (not isinstance(nb[-1], str) or not nb[-1].strip()):
                                    break
                            text = "\n".join([x for x in nb if isinstance(x, str)]).strip()
                            if text:
                                chunks.append({"id": f"sec{si}_blk{block_id}", "type": "narrative", "text": text})
                                block_id += 1
                    except Exception:
                        i += 1  # advance to avoid getting stuck
        except Exception as e:
            # logger.info(f"extract_ordered_chunks error: {e}")
            logger.info(f"extract_ordered_chunks error: {e}")
        return chunks

    def get_chunks(self):
        ids = 0
        table_ids = {}
        documents = []

        try:
            doc_items = self.docs.keys() if isinstance(self.docs, dict) else []
        except Exception:
            doc_items = []

        for document_type in doc_items:
            try:
                payload = self.docs.get(document_type, {})
                contents = payload.get("content", {}) if isinstance(payload, dict) else {}
                link = payload.get("link") if isinstance(payload, dict) else None
            except Exception:
                continue

            for key, value in contents.items() if isinstance(contents, dict) else []:
                try:
                    html_splits = self.split_html_pages(value)
                    splits = [convert_to_markdown(s) for s in html_splits]
                except Exception:
                    splits = []

                for split in splits:
                    try:
                        chunk_list = self.extract_ordered_chunks(split)
                    except Exception:
                        chunk_list = []

                    try:
                        text = ""
                        tables = ""
                        for k in chunk_list:
                            try:
                                if k.get("type") == "narrative":
                                    text += (k.get("text") or "") + "\n"
                                else:
                                    tables += (k.get("table") or "") + "\n"
                            except Exception:
                                continue

                        table_ids[str(ids)] = {"document_type": document_type, "url": link, "section": key, "tables": tables}
                        md = {"url": link, "section": key, "document_type": document_type, "id": str(ids)}
                        documents.append(Document(page_content=text, metadata=md))
                        ids += 1
                    except Exception:
                        continue

        try:
            chunk_docs = self.text_splitter.split_documents(documents)
        except Exception:
            chunk_docs = []

        for i, d in enumerate(chunk_docs):
            try:
                d.metadata["chunk_id"] = f"{d.metadata['document_type']}_{i}"
            except Exception:
                # ensure metadata exists
                try:
                    d.metadata = d.metadata if isinstance(d.metadata, dict) else {}
                    d.metadata["chunk_id"] = f"{d.metadata.get('document_type', 'doc')}_{i}"
                except Exception:
                    pass
            try:
                # append tables per original logic
                tid = d.metadata.get("id") if isinstance(d.metadata, dict) else None
                d.page_content += f"""\n{table_ids.get(tid, {}).get("tables", "")}"""
            except Exception as e:
                # logger.info(f"Append tables to page_content error: {e}")
                logger.info(f"Append tables to page_content error: {e}")

        return chunk_docs


async def get_sec_docs(ticker, api_key):
    logger.info("fetching docs")
    sec = SEC(ticker, api_key)
    docs, statements = await sec.aget_filings_data()
    logger.info("creating chunks")
    doc_processor = Chunks(docs)
    chunk_docs = doc_processor.get_chunks()
    logger.info("chunks created")
    return chunk_docs, statements
