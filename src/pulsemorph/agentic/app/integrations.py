"""Application memory and optional read-only Zetaris context retrieval.

Meterless is the real pinned local upstream engine. Zetaris is never reported
connected until an authenticated query succeeds. Environment credentials remain
in process memory; remote error bodies and headers are never logged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import threading
from datetime import datetime, timezone
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlsplit
import uuid

METERLESS_COMMIT = "1742dec56a66e58eef9dda2d2b865823758ca8cd"
ZETARIS_API_SOURCE = "https://github.com/zetaris/getting-started/blob/main/scripts/query_zetaris.py"
ZETARIS_AUTH_SOURCE = "https://github.com/zetaris/getting-started/blob/main/scripts/HOWTO.md"
_PRIVATE = re.compile(r"password|secret|token|api.?key|authorization|credential", re.I)
CAPABILITY_COLUMNS = ("source_id", "title", "body_id", "capability", "evidence_class", "license_scope")
CAPABILITY_LIMITS = {"source_id": 80, "title": 240, "body_id": 64, "capability": 12000,
                     "evidence_class": 120, "license_scope": 1000}
_SCOPE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}")
_STORE_LOCKS = {}
_STORE_LOCK_GUARD = threading.Lock()


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("Authenticated API redirects are disabled")


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items() if not _PRIVATE.search(str(k))}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _valid_context_text(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


class AppIntegrations:
    def __init__(self, root=None, artifact_root=None):
        self.root = Path(root or Path(__file__).resolve().parents[4]).resolve()
        self.artifact_root = Path(artifact_root or self.root / "artifacts/runs/agentic/app").resolve()
        allowed = self.root / "artifacts/runs/agentic"
        if not self.artifact_root.is_relative_to(allowed) or self.artifact_root == allowed:
            raise ValueError("App artifacts must be a dedicated directory under artifacts/runs/agentic")
        self.memory_dir = self.artifact_root / "meterless_world"
        with _STORE_LOCK_GUARD:
            self._lock = _STORE_LOCKS.setdefault(str(self.memory_dir), threading.RLock())
        self._meterless = {"state": "not_checked", "mode": "local_upstream", "commit": METERLESS_COMMIT}
        self._zetaris_verified = False
        self._zetaris_transport_verified = False
        self._zetaris_usable_context_verified = False

    def _bridge(self, operation, record=None):
        runner = self.root / "research/vendor/Meterless/engines/world-model/reference/node_modules/tsx/dist/cli.mjs"
        bridge = self.root / "experiments/agentic/app/meterless_bridge.ts"
        with self._lock:
            result = subprocess.run(["node", str(runner), str(bridge)], input=json.dumps({
                "operation": operation, "directory": str(self.memory_dir), "record": record}),
                text=True, encoding="utf-8", capture_output=True, cwd=self.root, timeout=40)
            if result.returncode:
                self._meterless["state"] = "error"
                raise RuntimeError("Meterless bridge failed; check pinned checkout and Node/tsx installation")
            payload = json.loads(result.stdout)
            self._meterless.update(state="available", persisted_reopen_equal=payload["persisted_reopen_equal"], event_count=payload["event_count"])
            return payload

    def record_job(self, job):
        return self._record("app_job", job)

    def record_trial(self, job_id, trial):
        return self._record("app_trial", dict(trial, job_id=str(job_id)))

    def _record(self, kind, value):
        record = _clean(dict(value))
        identity = record.get("id") or record.get("job_id" if kind == "app_job" else "trial_id")
        if not identity:
            raise ValueError("Memory record requires id")
        record.update(type=kind, id=str(identity), updated_at=datetime.now(timezone.utc).isoformat())
        self._bridge("record", record)
        return record

    def list_memory(self, limit=50):
        limit = max(1, min(int(limit), 500))
        records = self._bridge("list")["records"]
        return sorted(records, key=lambda item: item.get("updated_at", ""), reverse=True)[:limit]

    def status(self):
        required = ["ZETARIS_BASE_URL", "ZETARIS_ORG_ID", "ZETARIS_API_KEY", "ZETARIS_APP_CONTEXT_TABLE"]
        missing = [name for name in required if not os.environ.get(name)]
        transport_selected = os.environ.get("ZETARIS_APP_TRANSPORT") == "ui_proxy_experimental"
        return {"meterless": dict(self._meterless), "zetaris": {
            "state": "query_verified" if self._zetaris_verified else ("missing_configuration" if missing else (
                "experimental_transport_not_selected" if not transport_selected else "configured_unverified")),
            "missing": missing, "mode": "read_only_ui_proxy", "live_query_verified": self._zetaris_verified,
            "transport_http_verified": self._zetaris_transport_verified,
            "usable_context_verified": self._zetaris_usable_context_verified,
            "context_authority": "untrusted_capability_context_only",
            "transport_selection": "ZETARIS_APP_TRANSPORT=ui_proxy_experimental",
            "contract": "official_getting_started_helper; deployment_unverified_until_query",
            "required_table_columns": ["source_id", "title", "body_id", "capability", "evidence_class", "license_scope"],
            "api_reference": ZETARIS_API_SOURCE, "authentication_reference": ZETARIS_AUTH_SOURCE}}

    def query_sources(self, query, limit=8, source_id=None, body_id="g1"):
        """Return app evidence memory and optional remote capability records.

        Remote records are context only; they do not grant execution or rights.
        """
        limit = max(1, min(int(limit), 50))
        terms = str(query).casefold().split()
        local = self.list_memory(500)
        if source_id is not None:
            # Apply body/source scope BEFORE truncation. Successful measured
            # examples should not disappear merely because their prose uses
            # different words than the next instruction.
            local = [row for row in local if row.get("type") == "app_trial"
                     and row.get("source_id") == source_id and type(row.get("index")) is int and row["index"] >= 0
                     and row.get("metrics", {}).get("integrity_passed") is True]
            def rank(row):
                metrics = row["metrics"]
                words = sum(term in json.dumps(row, ensure_ascii=False).casefold() for term in terms)
                return (metrics.get("physical_gate_passed") is True and metrics.get("intent_passed") is True,
                        metrics.get("physical_gate_passed") is True, words, row.get("updated_at", ""))
            memory = sorted(local, key=rank, reverse=True)[:limit]
            policy = "source_first; verified successful examples, physical gate, text overlap, recency"
        else:
            scored = [(sum(term in json.dumps(row, ensure_ascii=False).casefold() for term in terms), row) for row in local]
            memory = [row for score, row in sorted(scored, key=lambda pair: pair[0], reverse=True) if score or not terms][:limit]
            policy = "text_overlap"
        remote = {"state": self.status()["zetaris"]["state"], "records": [], "executed": False,
                  "transport_http_verified": False, "query_verified": False, "usable_context_verified": False,
                  "context_authority": "untrusted_capability_context_only"}
        if not self.status()["zetaris"]["missing"] and os.environ.get("ZETARIS_APP_TRANSPORT") == "ui_proxy_experimental":
            remote = self._query_zetaris(str(query), limit, source_id=source_id, body_id=body_id)
        return {"query": str(query), "memory": memory, "zetaris": remote, "memory_policy": policy,
                "source_scope": source_id, "body_scope": body_id}

    def _query_zetaris(self, query, limit, source_id=None, body_id="g1"):
        self._zetaris_transport_verified = False
        self._zetaris_verified = False
        self._zetaris_usable_context_verified = False
        limit = max(1, min(int(limit), 50))
        def result(state, *, executed=False, http=False, verified=False, records=None, **extra):
            return {"state": state, "records": records or [], "executed": executed,
                    "transport_http_verified": http, "query_verified": verified,
                    "usable_context_verified": bool(records), "read_only": True,
                    "context_authority": "untrusted_capability_context_only",
                    "source_scope": source_id, "body_scope": body_id, **extra}
        if (not isinstance(body_id, str) or not _SCOPE_ID.fullmatch(body_id) or len(body_id) > 64
                or (source_id is not None and (not isinstance(source_id, str) or not _SCOPE_ID.fullmatch(source_id)))):
            return result("invalid_scope_configuration")
        if os.environ.get("ZETARIS_APP_TRANSPORT") != "ui_proxy_experimental":
            return result("experimental_transport_not_selected")
        required = ("ZETARIS_APP_CONTEXT_TABLE", "ZETARIS_BASE_URL", "ZETARIS_API_KEY", "ZETARIS_ORG_ID")
        if any(not os.environ.get(key) for key in required):
            return result("missing_configuration")
        table = os.environ["ZETARIS_APP_CONTEXT_TABLE"]
        base = os.environ["ZETARIS_BASE_URL"].rstrip("/")
        parsed = urlsplit(base)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,2}", table):
            return result("invalid_table_configuration")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
            return result("invalid_endpoint_configuration")
        if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            return result("invalid_endpoint_configuration")
        # Identifiers are fixed or table-validated. Scope values are data, never SQL.
        literal = lambda value: "'" + value.replace("'", "''") + "'"
        filters = ["body_id = " + literal(body_id)]
        if source_id is not None:
            filters.append("source_id = " + literal(source_id))
            lookup_policy = "exact_source_and_body"
        else:
            terms = re.findall(r"[a-z0-9]{2,32}", str(query).lower())[:3]
            if terms:
                filters.append("(" + " OR ".join("LOWER(capability) LIKE " + literal("%" + term + "%") for term in terms) + ")")
            lookup_policy = "body_and_at_most_three_short_terms"
        sql = f"SELECT {','.join(CAPABILITY_COLUMNS)} FROM {table} WHERE {' AND '.join(filters)} LIMIT {limit}"
        body = {"sql": sql, "limit": limit}
        if os.environ.get("ZETARIS_ENGINE_ID"):
            body["engineId"] = os.environ["ZETARIS_ENGINE_ID"]
        request = Request(base + "/api/proxy/sql-editor/sqls/run-query", data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + os.environ["ZETARIS_API_KEY"],
            "X-Org-ID": os.environ["ZETARIS_ORG_ID"], "X-Request-ID": str(uuid.uuid4())})
        try:
            with build_opener(_NoRedirect()).open(request, timeout=15) as response:
                raw = response.read(2_000_001)
                self._zetaris_transport_verified = True
                if len(raw) > 2_000_000:
                    raise ValueError("Capability response exceeds bounded size")
                payload = json.loads(raw)
            self._zetaris_transport_verified = True
            if not isinstance(payload, (dict, list)) or (isinstance(payload, dict) and (
                payload.get("error") or payload.get("errors") or payload.get("success") is False
                or str(payload.get("status", "")).lower() in {"error", "failed"}
            )):
                raise RuntimeError("API returned an unsuccessful query response")
            rows = payload if isinstance(payload, list) else next((payload[key] for key in ("rows", "data", "results") if isinstance(payload.get(key), list)), None)
            if rows is None:
                return result("response_schema_unverified", executed=True, http=True)
            records, rejected = [], 0
            for row in rows[:500]:
                if (not isinstance(row, dict) or set(row) != set(CAPABILITY_COLUMNS)
                        or any(not _valid_context_text(row[key], CAPABILITY_LIMITS[key]) for key in CAPABILITY_COLUMNS)):
                    rejected += 1
                    continue
                if row["body_id"] != body_id or (source_id is not None and row["source_id"] != source_id):
                    rejected += 1
                    continue
                if not _SCOPE_ID.fullmatch(row["source_id"]):
                    rejected += 1
                    continue
                if len(records) < limit:
                    records.append({key: row[key].strip() for key in CAPABILITY_COLUMNS})
            rejected += max(0, len(rows) - 500)
            self._zetaris_verified = True
            self._zetaris_usable_context_verified = bool(records)
            return result("query_verified", executed=True, http=True, verified=True, records=records,
                          accepted_count=len(records), rejected_count=rejected,
                          retrieval_policy=lookup_policy, capability_field_limits=CAPABILITY_LIMITS)
        except Exception:
            self._zetaris_verified = False
            self._zetaris_usable_context_verified = False
            return result("query_failed", executed=True, http=self._zetaris_transport_verified,
                          error="Zetaris read-only query failed; verify endpoint, authorization, engine and context table")
