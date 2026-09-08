"""
The Streamlit console's client for the BankLens API.

When BANKLENS_API_URL is set, main.py stops importing the pipeline and talks
to the API instead: login, upload, read metrics, ask for a profile, stream
chat. Everything the views render is rebuilt from API JSON into the same
objects the direct mode uses (FinancialMetrics, CustomerProfile, a
categorized DataFrame), so the views do not know which mode they are in.
"""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx
import pandas as pd

from app.pipeline.agent import CustomerProfile
from app.pipeline.analyzer import FinancialMetrics


class ApiError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(f"{status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class BankLensClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    # ── plumbing ─────────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def _raise(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(response.status_code, str(detail))

    def _get(self, path: str) -> dict | list:
        response = httpx.get(
            f"{self.base_url}{path}", headers=self._headers(), timeout=self.timeout
        )
        self._raise(response)
        return response.json()

    def _post(self, path: str, **kwargs) -> dict:
        response = httpx.post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            timeout=self.timeout,
            **kwargs,
        )
        self._raise(response)
        return response.json()

    # ── auth ─────────────────────────────────────────────────────────────────

    def login(self, tenant: str, email: str, password: str) -> dict:
        body = self._post(
            "/auth/login", json={"tenant": tenant, "email": email, "password": password}
        )
        self.token = body["access_token"]
        return body

    def me(self) -> dict:
        return self._get("/auth/me")

    def health(self) -> dict:
        return self._get("/health")

    # ── customers & statements ───────────────────────────────────────────────

    def customers(self) -> list[dict]:
        return self._get("/customers")

    def statements(self) -> list[dict]:
        return self._get("/statements")

    def upload_statement(self, customer_id: str, filename: str, content: bytes) -> dict:
        return self._post(
            "/statements",
            data={"customer_id": customer_id},
            files={"file": (filename, content)},
        )

    def statement(self, statement_id: str) -> dict:
        return self._get(f"/statements/{statement_id}")

    def generate_profile(self, statement_id: str) -> dict:
        return self._post(f"/statements/{statement_id}/profile")

    def latest_profile(self, statement_id: str) -> dict | None:
        try:
            return self._get(f"/statements/{statement_id}/profile")
        except ApiError as exc:
            if exc.status_code == 404:
                return None
            raise

    # ── decision graph ───────────────────────────────────────────────────────

    def _sse(self, method: str, path: str, json_body=None) -> Iterator[dict]:
        """Yield SSE events as dicts: {"event": kind, ...payload}."""
        with httpx.stream(
            method,
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=json_body,
            timeout=self.timeout,
        ) as response:
            self._raise(response)
            kind = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    kind = line[7:].strip()
                elif line.startswith("data: "):
                    payload = json.loads(line[6:])
                    body = payload if isinstance(payload, dict) else {"data": payload}
                    yield {"event": kind, **body}

    def run_graph(self, statement_id: str) -> Iterator[dict]:
        return self._sse("POST", f"/statements/{statement_id}/run")

    def runs(self, statement_id: str) -> list[dict]:
        return self._get(f"/statements/{statement_id}/runs")

    def audit(self, statement_id: str) -> list[dict]:
        return self._get(f"/statements/{statement_id}/audit")

    def gateway(self) -> dict:
        return self._get("/platform/gateway")

    def jobs_summary(self) -> dict:
        return self._get("/jobs/summary")

    def traces(self, statement_id: str) -> list[dict]:
        return self._get(f"/statements/{statement_id}/traces")

    def trace(self, trace_id: str) -> dict:
        return self._get(f"/traces/{trace_id}")

    def reviews(self, state: str = "pending") -> list[dict]:
        return self._get(f"/reviews?state={state}")

    def decide(self, decision_id: str, action: str, note: str | None) -> Iterator[dict]:
        return self._sse(
            "POST", f"/reviews/{decision_id}", {"action": action, "note": note}
        )

    # ── chat (SSE) ───────────────────────────────────────────────────────────

    def chat(
        self, statement_id: str, question: str, history: list[dict]
    ) -> Iterator[str]:
        """Yield answer tokens as they arrive over Server-Sent Events."""
        with httpx.stream(
            "POST",
            f"{self.base_url}/statements/{statement_id}/chat",
            headers=self._headers(),
            json={"question": question, "history": history},
            timeout=self.timeout,
        ) as response:
            self._raise(response)
            event = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    event = line[7:].strip()
                elif line.startswith("data: "):
                    data = json.loads(line[6:])
                    if event == "token":
                        yield data
                    elif event in ("blocked", "abstained"):
                        label = (
                            "🛡️ blocked" if event == "blocked" else "🤷 out of scope"
                        )
                        yield f"*{label}: {data.get('reason', event)}*  \n"
                    elif event == "error":
                        raise ApiError(500, data)
                    # "done" carries the full answer; tokens already covered it.


# ── Adapters from API JSON to the objects the views expect ──────────────────


def metrics_from_detail(detail: dict) -> FinancialMetrics:
    return FinancialMetrics.model_validate(detail["metrics"])


def dataframe_from_detail(detail: dict) -> pd.DataFrame:
    df = pd.DataFrame(detail["transactions"])
    if df.empty:
        return pd.DataFrame(
            columns=["date", "description", "amount", "type", "category"]
        )
    return df


def profile_from_response(body: dict, tenant: str) -> CustomerProfile:
    from app.core.context import tenant_scope

    profile = dict(body["profile"])
    profile.setdefault("retrieved_sources", body.get("retrieved_sources", []))
    # Product-name validation needs the tenant's catalogue in scope.
    with tenant_scope(tenant):
        return CustomerProfile.model_validate(profile)
