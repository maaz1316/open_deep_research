from datetime import datetime
import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import openai
import requests

from infras.secrets.constants import SecretKeys
from libs.secrets.secrets import Secrets

from .helpers import company_translation, logger


class CapIQConnector:
    """CapIQ API connector for company data retrieval"""

    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=Secrets.get(SecretKeys.OPENAI_API_KEY))
        self.logger = logger
        # CORRECT endpoint for data requests
        self.base_url = "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest/v3/clientservice.json"

        # Authentication endpoint (different from data endpoint)
        self.auth_url = "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest/authenticate/api/v1/token"

        # Try to get credentials from Secrets first, then fall back to environment variables

        try:
            self.username = Secrets.get(SecretKeys.CAPIQ_USERNAME)
        except BaseException:
            self.username = os.getenv("CAPIQ_USERNAME", "")

        try:
            self.password = Secrets.get(SecretKeys.CAPIQ_PASSWORD)
        except BaseException:
            self.password = os.getenv("CAPIQ_PASSWORD", "")

        if not self.username or not self.password:
            raise ValueError("CapIQ credentials not found in secrets or environment variables")

        # Get bearer token on initialization
        self.bearer_token = self._get_bearer_token()

    def _get_bearer_token(self) -> str:
        """Get bearer token for authentication"""
        try:
            headers = {"Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}

            data = {"username": self.username, "password": self.password}

            self.logger.info("Requesting bearer token from CapIQ")
            response = requests.post(self.auth_url, headers=headers, data=data, timeout=30)

            if response.status_code != 200:
                raise Exception(f"Token authentication failed: {response.text}")

            token_data = response.json()
            access_token = token_data.get("access_token")

            if not access_token:
                raise Exception("No access token in response")

            self.logger.info("Successfully obtained bearer token")
            return access_token

        except Exception as e:
            self.logger.error(f"Error getting bearer token: {e}")
            raise

    def _make_request(self, input_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make authenticated request to CapIQ API using bearer token"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.bearer_token}",  # Bearer token in header
            }

            # NO username/password in payload anymore
            payload = {"inputRequests": input_requests}

            self.logger.info(f"Making CapIQ API request with {len(input_requests)} requests")
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)

            if response.status_code != 200:
                self.logger.error(f"CapIQ API error: {response.status_code} - {response.text}")
                raise Exception(f"CapIQ API request failed: {response.text}")

            return response.json()

        except requests.exceptions.Timeout:
            raise Exception("CapIQ API request timed out")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"CapIQ API request error: {e}")
            raise Exception(f"Failed to connect to CapIQ API: {str(e)}")

    # Task 1.1: Parse and normalize company URL/name input
    def parse_company_input(self, input_str: str) -> Dict[str, Any]:
        """
        Parse and normalize company URL or name input

        Args:
            input_str: Company URL or name

        Returns:
            dict: Parsed data with domain/name and input type
        """
        try:
            input_str = input_str.strip()

            # Check if input is a URL
            url_patterns = [r"^https?://", r"^www\.", r"\.(com|org|net|io|ai|co|edu|gov)(/|$)"]

            is_url = any(re.search(pattern, input_str, re.IGNORECASE) for pattern in url_patterns)

            if is_url:
                # Normalize URL
                if not input_str.startswith(("http://", "https://")):
                    input_str = "https://" + input_str

                parsed_url = urlparse(input_str)
                domain = parsed_url.netloc or parsed_url.path

                # Remove www. prefix
                domain = re.sub(r"^www\.", "", domain, flags=re.IGNORECASE)

                # Extract main domain (remove subdomains for common cases)
                domain_parts = domain.split(".")
                if len(domain_parts) > 2:
                    # Keep last two parts (domain + TLD)
                    domain = ".".join(domain_parts[-2:])

                self.logger.info(f"Parsed URL '{input_str}' to domain '{domain}'")

                return {"input_type": "url", "original_input": input_str, "domain": domain, "search_term": domain}
            else:
                # Input is a company name
                self.logger.info(f"Input '{input_str}' identified as company name")

                return {"input_type": "name", "original_input": input_str, "company_name": input_str, "search_term": input_str}

        except Exception as e:
            self.logger.error(f"Error parsing company input: {e}")
            raise ValueError(f"Invalid company input: {str(e)}")

    # Task 1.2: Resolve company identity via CapIQ API
    def resolve_company_identity(self, parsed_input: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Resolve company identity using CapIQ API (two-step process)
        Step 1: Get company IDs using GDSHE with IQ_COMPANY_ID_QUICK_MATCH
        Step 2: Get websites for those IDs and match
        """
        try:
            search_term = parsed_input.get("search_term")
            input_type = parsed_input.get("input_type")

            # Step 1: Get company IDs using quick match
            input_requests = [
                {
                    "function": "GDSHE",
                    "identifier": search_term,
                    "mnemonic": "IQ_COMPANY_ID_QUICK_MATCH",
                    "properties": {"startRank": "1", "endRank": "10"},  # Get top 10 matches
                }
            ]

            self.logger.info(f"Searching CapIQ for company IDs: {search_term}")
            result = self._make_request(input_requests)

            # Parse company IDs from response
            gds_response = result.get("GDSSDKResponse", [])
            if not gds_response or gds_response[0].get("ErrMsg"):
                self.logger.warning(f"No companies found in CapIQ for: {search_term}")
                return None

            rows = gds_response[0].get("Rows", [])
            if not rows:
                return None

            company_ids = [row["Row"][0] for row in rows if row.get("Row")]

            # Step 2: Get websites for all company IDs
            if input_type == "url":
                matched_company_id = self._match_company_by_website(company_ids, parsed_input.get("domain"))
            else:
                # For name search, take the first result
                matched_company_id = company_ids[0]

            if not matched_company_id:
                self.logger.warning(f"No matching company found for: {search_term}")
                return None

            return matched_company_id

        except Exception as e:
            self.logger.error(f"Error resolving company identity: {e}")
            return None


class CapIQError(Exception):
    """Custom exception for Capital IQ API errors."""

    pass


class CapIQFinancialsClient:
    """
    Retrieve historical financials, cashflows, ratios, capital structure,
    and transaction data (M&A, buybacks, IPOs, etc.) from S&P Capital IQ.
    """

    BASE_URL = "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest/v3/clientservice.json"
    AUTH_URL = "https://api-ciq.marketintelligence.spglobal.com/gdsapi/rest/authenticate/api/v1/token"

    # -----------------------------------------------------------------------
    # Transaction Mnemonics Mapping
    # -----------------------------------------------------------------------
    TRANSACTION_FIELD_MAP = {
        "IQ_TR_TRANSACTION_TYPE": "Transaction Type",
        "IQ_TR_ANN_DATE_BL": "Public Announced Date",
        "IQ_TR_TOTALVALUE": "Transaction Size",
        "IQ_TR_TARGETNAME": "Target / Issuer Name",
        "IQ_TR_BNKY_ADVISOR_NAME_LIST": "Advisor Name(s)",
        "IQ_TR_PCT_SOUGHT_ACQUIRED_FINAL": "Percent Sought / Acquired",
        "IQ_TR_CURRENCY": "Transaction Currency",
    }

    TRANSACTION_LIST_MNEMONICS = [
        "IQ_TRANSACTION_LIST",
        "IQ_TRANSACTION_LIST_BANKRUPTCY",
        "IQ_TRANSACTION_LIST_BUYBACK",
        "IQ_TRANSACTION_LIST_MA",
        "IQ_TRANSACTION_LIST_PP",
        "IQ_TRANSACTION_LIST_PO",
    ]

    # -----------------------------------------------------------------------
    # Authentication
    # -----------------------------------------------------------------------
    def __init__(self):

        try:
            self.username = Secrets.get(SecretKeys.CAPIQ_USERNAME)
        except BaseException:
            self.username = os.getenv("CAPIQ_USERNAME", "")

        try:
            self.password = Secrets.get(SecretKeys.CAPIQ_PASSWORD)
        except BaseException:
            self.password = os.getenv("CAPIQ_PASSWORD", "")

        if not self.username or not self.password:
            raise ValueError("CapIQ credentials not found in secrets or environment variables")

        self.session = requests.Session()
        self.token = self._authenticate()

    def _authenticate(self) -> str:
        """Obtain an OAuth bearer token for the Capital IQ API."""
        headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
        payload = {"username": self.username, "password": self.password}
        response = self.session.post(self.AUTH_URL, data=payload, headers=headers)
        if response.status_code != 200:
            raise CapIQError(f"Auth failed: {response.status_code} - {response.text}")

        try:
            data = response.json()
        except json.JSONDecodeError:
            raise CapIQError("Invalid JSON response during authentication.")

        token = data.get("access_token") or data.get("token")
        if not token:
            raise CapIQError("Token missing in authentication response.")
        return token

    # -----------------------------------------------------------------------
    # Core API invocation
    # -----------------------------------------------------------------------
    def _invoke(self, body: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        response = self.session.post(self.BASE_URL, headers=headers, json=body)
        if response.status_code != 200:
            raise CapIQError(f"Request failed: {response.status_code} - {response.text}")

        try:
            return response.json()
        except json.JSONDecodeError:
            raise CapIQError(f"Invalid JSON returned:\n{response.text[:300]}")

    # -----------------------------------------------------------------------
    # Transaction Helpers
    # -----------------------------------------------------------------------
    def _extract_transaction_ids(self, raw: dict) -> list[str]:
        """Extract all transaction IDs from the GDSP transaction list response."""
        ids = []
        for item in raw.get("GDSSDKResponse", []):
            rows = item.get("Rows", [])
            if not rows:
                continue
            val = rows[0].get("Row", [None])[0]
            if val and val.startswith("IQTR"):
                ids.append(val)
        return list(set(ids))

    def _fetch_transaction_details(self, transaction_id: str) -> dict:
        """Fetch detailed information for a given Transaction ID (IQTR...)."""
        body = {
            "inputRequests": [{"function": "GDSP", "identifier": transaction_id, "mnemonic": m} for m in self.TRANSACTION_FIELD_MAP.keys()]
        }
        raw = self._invoke(body)

        result = {"transaction_id": transaction_id}
        for item in raw.get("GDSSDKResponse", []):
            mnemonic = item.get("Mnemonic")
            rows = item.get("Rows", [])
            if not rows:
                continue
            val = rows[0].get("Row", [None])[0]
            if val and mnemonic in self.TRANSACTION_FIELD_MAP:
                readable_key = self.TRANSACTION_FIELD_MAP[mnemonic]
                result[readable_key] = val
        return result

    def _fetch_transactions(self, company_id: str) -> list[dict]:
        """Retrieve all transactions (M&A, Buyback, etc.) for a given company."""
        # Step 1: Request transaction lists
        body = {"inputRequests": [{"function": "GDSP", "identifier": company_id, "mnemonic": m} for m in self.TRANSACTION_LIST_MNEMONICS]}
        raw = self._invoke(body)
        transaction_ids = self._extract_transaction_ids(raw)

        # Step 2: Fetch details for each transaction
        transactions = []
        for tid in transaction_ids:
            try:
                tx = self._fetch_transaction_details(tid)
                transactions.append(tx)
            except Exception as e:
                transactions.append({"transaction_id": tid, "error": str(e)})
        return transactions

    # -----------------------------------------------------------------------
    # Financial Parsing
    # -----------------------------------------------------------------------
    def _parse_financial_response(self, raw: dict) -> dict:
        parsed = {
            "income_statement": {},
            "balance_sheet": {},
            "cashflow": {},
            "financial_ratios": {},
            "capital_structure": {},
        }
        for item in raw.get("GDSSDKResponse", []):
            mnemonic = item.get("Mnemonic")
            if not mnemonic:
                continue

            rows = item.get("Rows", [])
            section = self._classify_section(mnemonic)
            if not section:
                continue

            parsed_section = parsed[section].setdefault(mnemonic, [])
            for r in rows:
                values = r.get("Row", [])
                if len(values) >= 2:
                    parsed_section.append({"period": values[1], "value": values[0]})
        return parsed

    @staticmethod
    def _classify_section(mnemonic: str) -> str | None:
        if mnemonic in ["IQ_TOTAL_REV", "IQ_NI", "IQ_EBIT", "IQ_EBITA", "IQ_DILUT_EPS_EXCL", "IQ_GP", "IQ_CASH_OPER_AP"]:
            return "income_statement"
        if mnemonic in ["IQ_TOTAL_ASSETS", "IQ_TOTAL_EQUITY", "IQ_TOTAL_DEBT", "IQ_TOTAL_LIAB"]:
            return "balance_sheet"
        if mnemonic in ["IQ_CAPEX", "IQ_CASH_OPER"]:
            return "cashflow"
        if mnemonic in [
            "IQ_RETURN_CAPITAL",
            "IQ_TOTAL_DEBT_EQUITY",
            "IQ_TOTAL_DEBT_CAPITAL",
            "IQ_LT_DEBT_EQUITY",
            "IQ_CURRENT_RATIO",
            "IQ_QUICK_RATIO",
            "IQ_EBITDA_MARGIN",
            "IQ_NI_MARGIN",
            "IQ_GROSS_MARGIN",
        ]:
            return "financial_ratios"
        if mnemonic in ["IQ_TOTAL_DEBT_EBITDA_CAPEX", "IQ_NET_DEBT_EBITDA_CAPEX", "IQ_NET_DEBT_EBITDA", "IQ_TOTAL_DEBT_EBITDA"]:
            return "capital_structure"
        return None

    # -----------------------------------------------------------------------
    # Unified Financials + Transactions
    # -----------------------------------------------------------------------
    def fetch_all_financials(self, company_id: str) -> dict:
        """Retrieve comprehensive financials + transactions for a given company."""
        now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        body = {"inputRequests": []}

        # Financials
        for m in ["IQ_TOTAL_REV", "IQ_NI", "IQ_EBIT", "IQ_EBITA", "IQ_DILUT_EPS_EXCL", "IQ_GP"]:
            body["inputRequests"].append(
                {
                    "function": "GDSHE",
                    "identifier": company_id,
                    "mnemonic": m,
                    "properties": {"periodType": "IQ_FY-5", "metadataTag": "PeriodDate"},
                }
            )
        for m in ["IQ_TOTAL_ASSETS", "IQ_TOTAL_EQUITY", "IQ_TOTAL_DEBT", "IQ_TOTAL_LIAB"]:
            body["inputRequests"].append(
                {
                    "function": "GDSHE",
                    "identifier": company_id,
                    "mnemonic": m,
                    "properties": {"periodType": "IQ_FQ-12", "metadataTag": "PeriodDate"},
                }
            )
        for m in ["IQ_CAPEX", "IQ_CASH_OPER"]:
            body["inputRequests"].extend(
                [
                    {
                        "function": "GDSHE",
                        "identifier": company_id,
                        "mnemonic": m,
                        "properties": {"periodType": "IQ_FY-5", "metadataTag": "PeriodDate"},
                    },
                    {
                        "function": "GDSHE",
                        "identifier": company_id,
                        "mnemonic": m,
                        "properties": {"periodType": "IQ_FQ-4", "metadataTag": "PeriodDate"},
                    },
                ]
            )
        for m in [
            "IQ_RETURN_CAPITAL",
            "IQ_TOTAL_DEBT_EQUITY",
            "IQ_TOTAL_DEBT_CAPITAL",
            "IQ_LT_DEBT_EQUITY",
            "IQ_CURRENT_RATIO",
            "IQ_QUICK_RATIO",
            "IQ_EBITDA_MARGIN",
            "IQ_NI_MARGIN",
            "IQ_GROSS_MARGIN",
        ]:
            body["inputRequests"].append(
                {
                    "function": "GDSHE",
                    "identifier": company_id,
                    "mnemonic": m,
                    "properties": {"periodType": "IQ_FY-5", "metadataTag": "PeriodDate"},
                }
            )
        for m in ["IQ_TOTAL_DEBT_EBITDA_CAPEX", "IQ_NET_DEBT_EBITDA_CAPEX", "IQ_NET_DEBT_EBITDA", "IQ_TOTAL_DEBT_EBITDA"]:
            body["inputRequests"].append(
                {
                    "function": "GDSHE",
                    "identifier": company_id,
                    "mnemonic": m,
                    "properties": {"periodType": "IQ_FY-5", "metadataTag": "PeriodDate"},
                }
            )

        # Filing currency + description
        for m in ["IQ_FILING_CURRENCY", "IQ_DESCRIPTION_LONG"]:
            body["inputRequests"].append({"function": "GDSP", "identifier": company_id, "mnemonic": m})

        raw = self._invoke(body)
        parsed = self._parse_financial_response(raw)

        # Extract metadata
        filing_currency = "Unknown"
        business_description = None
        for item in raw.get("GDSSDKResponse", []):
            mnemonic = item.get("Mnemonic")
            rows = item.get("Rows", [])
            if not rows:
                continue
            value = rows[0].get("Row", [None])[0]
            if mnemonic == "IQ_FILING_CURRENCY" and value:
                filing_currency = value
            elif mnemonic == "IQ_DESCRIPTION_LONG" and value:
                business_description = value

        # Fetch transactions and merge
        transactions = self._fetch_transactions(company_id)

        return {
            "metadata": {
                "identifier": company_id,
                "generated_at": now,
                "source": "S&P Capital IQ GDS API",
                "filing_currency": filing_currency,
                "unit_scale": "Thousands",
                "business_description": business_description,
            },
            "data": parsed,
            "transactions": transactions,
        }


def get_capiq_data(company_name):

    connector = CapIQConnector()
    try:
        company_name_tr = company_translation(company_name)
    except Exception:
        company_name_tr = company_name
    parsed_input = connector.parse_company_input(company_name_tr)
    company_info = connector.resolve_company_identity(parsed_input)

    client = CapIQFinancialsClient()
    result = client.fetch_all_financials(company_info)

    return "*********CapIQ Data**********" + json.dumps(result) + "\n\nSOURCE: [CapIQ](None)"
