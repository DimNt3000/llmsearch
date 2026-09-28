# -*- coding: utf-8 -*-
"""Offline test harness for `llmsearch serve`, the mobile app's JSON API.

Run it with:

    python tests/test_server.py

It starts the real server on an ephemeral localhost port against a
temporary index, then exercises every endpoint over HTTP. No network and
no LLM calls: model output is faked in-process, and the crawl tests crawl
a throwaway site served from a temporary directory on localhost.

Exits 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

SCRATCH = Path(tempfile.mkdtemp(prefix="llmsearch-server-tests-"))
os.environ["LLMSEARCH_DATA_DIR"] = str(SCRATCH / "data")

import llmsearch.config as config  # noqa: E402
import llmsearch.indexer as indexer  # noqa: E402
import llmsearch.llm as llm  # noqa: E402
import llmsearch.server as server  # noqa: E402
import llmsearch.service as service  # noqa: E402

PASS: list[tuple[str, str]] = []
FAIL: list[tuple[str, str]] = []

REAL_CONFIG_PATH = config.CONFIG_PATH
HAD_USER_CONFIG = REAL_CONFIG_PATH.exists()
TEST_CONFIG = SCRATCH / "config.json"
config.CONFIG_PATH = TEST_CONFIG
TOKEN = "test-token-123"


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append((name, detail))
    print(("PASS " if cond else "FAIL ") + name + ("  | " + detail if detail and not cond else ""))


def T(name: str, fn) -> None:
    try:
        fn()
    except Exception:
        FAIL.append((name, "test raised"))
        print(f"FAIL {name}  | test raised:\n{traceback.format_exc(limit=4)}")


def set_config(**values) -> None:
    TEST_CONFIG.write_text(json.dumps(values), encoding="utf-8")


def call(method: str, path: str, body=None, token: str | None = TOKEN,
         raw: bytes | None = None, headers: dict | None = None, timeout: float = 30):
    """HTTP call returning (status, parsed_json_or_None, response_headers)."""
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
            status, hdrs = resp.status, dict(resp.headers)
    except urllib.error.HTTPError as err:
        payload, status, hdrs = err.read(), err.code, dict(err.headers)
    return status, (json.loads(payload) if payload else None), hdrs


def err_code(payload) -> str | None:
    return (payload or {}).get("error", {}).get("code")


# ---------------------------------------------------------------- fixtures
DOCS = SCRATCH / "docs"
DOCS.mkdir(parents=True)
_fixture_cfg = dict(config.DEFAULTS)
with indexer.open_db(config.db_path()) as _conn:
    indexer.add_document(_conn, "file:///apples.md", "apples",
                         "Apples grow on trees. The zylophant fruit is rare.", "file", _fixture_cfg)
    indexer.add_document(_conn, "file:///bananas.md", "bananas",
                         "Bananas are yellow tropical fruit grown in warm climates.", "file",
                         _fixture_cfg)
_conn.close()

# A tiny three-page site for crawl tests, served from a temporary directory.
SITE = SCRATCH / "site"
SITE.mkdir()
(SITE / "index.html").write_text(
    "<html><head><title>Crawl Home</title></head><body><p>Home page about quokkas.</p>"
    "<a href='page2.html'>two</a> <a href='page3.html'>three</a></body></html>",
    encoding="utf-8")
(SITE / "page2.html").write_text(
    "<html><head><title>Page Two</title></head><body><p>The vexillology section lives "
    "here.</p></body></html>", encoding="utf-8")
(SITE / "page3.html").write_text(
    "<html><head><title>Page Three</title></head><body><p>Third page on ornithology.</p>"
    "</body></html>", encoding="utf-8")


class _QuietSite(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SITE), **kwargs)

    def log_message(self, *args):
        pass


site = ThreadingHTTPServer(("127.0.0.1", 0), _QuietSite)
threading.Thread(target=site.serve_forever, daemon=True).start()
SITE_URL = f"http://127.0.0.1:{site.server_address[1]}"

set_config()
api = server.make_server("127.0.0.1", 0, TOKEN, log=lambda line: None)
threading.Thread(target=api.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{api.server_address[1]}"


# ---------------------------------------------------------------- 1. token
def t_token():
    first = server.load_or_create_token()
    check("token: created and persisted in the data dir",
          len(first) >= 20 and (config.data_dir() / server.TOKEN_FILE).is_file())
    check("token: reused across restarts", server.load_or_create_token() == first)
    rotated = server.load_or_create_token(rotate=True)
    check("token: --new-token rotates it", rotated != first and len(rotated) >= 20)
T("token", t_token)


# ---------------------------------------------------------------- 2. auth + plumbing
def t_auth():
    status, body, _ = call("GET", "/api/health", token=None)
    check("health: public, no token needed",
          status == 200 and body.get("ok") is True and body.get("service") == "llmsearch",
          f"{status} {body}")
    status, body, _ = call("GET", "/api/stats", token=None)
    check("stats without a token -> 401", status == 401 and err_code(body) == "unauthorized",
          f"{status} {body}")
    status, body, _ = call("GET", "/api/stats", token="wrong-token")
    check("stats with a wrong token -> 401", status == 401, f"{status}")
    status, body, _ = call("GET", "/api/stats", headers={"Authorization": "Basic abc"}, token=None)
    check("non-bearer auth scheme -> 401", status == 401, f"{status}")
    status, body, _ = call("GET", "/api/stats")
    check("stats with the token -> 200 with counts",
          status == 200 and body.get("documents") == 2 and "backend" in body, f"{status} {body}")
    status, body, _ = call("GET", "/api/nope")
    check("unknown path -> 404", status == 404 and err_code(body) == "not_found", f"{status}")
    status, body, _ = call("GET", "/api/ask")
    check("wrong method -> 405", status == 405 and err_code(body) == "method_not_allowed",
          f"{status}")
    status, _, hdrs = call("OPTIONS", "/api/search", token=None)
    check("CORS preflight -> 204 with allow headers",
          status == 204 and hdrs.get("Access-Control-Allow-Origin") == "*"
          and "Authorization" in hdrs.get("Access-Control-Allow-Headers", ""), f"{status} {hdrs}")
    _, _, hdrs = call("GET", "/api/health", token=None)
    check("server banner does not leak the Python version",
          "Python" not in hdrs.get("Server", ""), hdrs.get("Server", ""))
T("auth", t_auth)


# ---------------------------------------------------------------- 3. search
def t_search():
    status, body, _ = call("GET", "/api/search?q=zylophant")
    top = (body or {}).get("results", [{}])[0]
    check("search: finds the right document",
          status == 200 and top.get("title") == "apples", f"{status} {body}")
    check("search: result has snippet, text, and score",
          "zylophant" in top.get("snippet", "") and top.get("text") and top.get("score", 0) > 0,
          str(top))
    check("search: internal ids are not exposed",
          "chunk_id" not in top and "doc_id" not in top, str(top))
    status, body, _ = call("GET", "/api/search")
    check("search without q -> 400", status == 400 and err_code(body) == "bad_request",
          f"{status}")
    status, body, _ = call("GET", "/api/search?q=" + "x" * 1200)
    check("search with an oversized query -> 400", status == 400, f"{status}")
    status, body, _ = call("GET", "/api/search?q=%CE%BA%CE%B1%CE%BB%CE%B7%CE%BC%CE%AD%CF%81%CE%B1")
    check("search: non-ASCII query is handled", status == 200, f"{status}")
    set_config(backend="off")
    status, body, _ = call("GET", "/api/search?q=zylophant&smart=1&rerank=1")
    check("search --smart with no LLM degrades to BM25 with a notice",
          status == 200 and body.get("notice") and body["results"], f"{status} {body}")
    set_config()
T("search", t_search)


# ---------------------------------------------------------------- 4. LLM endpoints
def t_llm_endpoints():
    set_config(backend="off")
    status, body, _ = call("POST", "/api/ask", {"question": "zylophant"})
    check("ask with no LLM backend -> 503 llm_unavailable",
          status == 503 and err_code(body) == "llm_unavailable", f"{status} {body}")
    set_config()

    status, body, _ = call("POST", "/api/ask", {"question": "qqqqzzzz nothing matches"})
    check("ask with nothing relevant -> 422 no_context",
          status == 422 and err_code(body) == "no_context", f"{status} {body}")
    status, body, _ = call("POST", "/api/ask", {})
    check("ask without a question -> 400", status == 400, f"{status}")
    status, body, _ = call("POST", "/api/ask", {"question": 42})
    check("ask with a non-string question -> 400", status == 400, f"{status}")
    status, body, _ = call("POST", "/api/ask", raw=b"{not json")
    check("malformed JSON -> 400 bad_json", status == 400 and err_code(body) == "bad_json",
          f"{status} {body}")
    status, body, _ = call("POST", "/api/ask", raw=b"[1, 2]")
    check("JSON that is not an object -> 400", status == 400, f"{status}")
    status, body, _ = call("POST", "/api/ask", raw=b"x" * (server.MAX_BODY_BYTES + 10))
    check("oversized body -> 413", status == 413 and err_code(body) == "payload_too_large",
          f"{status}")

    real_complete = llm.complete
    try:
        llm.complete = lambda *a, **k: "Zylophants are rare fruit [1]."
        status, body, _ = call("POST", "/api/ask", {"question": "what is a zylophant?"})
        check("ask: answer with numbered sources",
              status == 200 and body.get("answer") == "Zylophants are rare fruit [1]."
              and body["sources"][0]["n"] == 1 and body["sources"][0]["title"] == "apples",
              f"{status} {body}")
        status, body, _ = call("POST", "/api/summarize", {"query": "fruit"})
        check("summarize: summary with sources",
              status == 200 and body.get("summary") and len(body.get("sources", [])) == 2,
              f"{status} {body}")

        def fail(*a, **k):
            raise llm.LLMError("model exploded")
        llm.complete = fail
        status, body, _ = call("POST", "/api/ask", {"question": "zylophant"})
        check("LLM failure -> 502 llm_error with its message",
              status == 502 and err_code(body) == "llm_error" and "exploded" in body["error"]["message"],
              f"{status} {body}")
    finally:
        llm.complete = real_complete

    status, body, _ = call("GET", "/api/web")
    check("web without q -> 400 (validated before any network call)", status == 400, f"{status}")
    status, body, _ = call("GET", "/api/web?q=x&n=99")
    check("web with out-of-range n -> 400", status == 400, f"{status}")
    status, body, _ = call("POST", "/api/web/ask", {})
    check("web/ask without a query -> 400", status == 400, f"{status}")
T("llm_endpoints", t_llm_endpoints)


def t_internal_errors():
    real_search = service.search
    try:
        def boom(*a, **k):
            raise RuntimeError("secret internal detail")
        service.search = boom
        stderr, sys.stderr = sys.stderr, open(os.devnull, "w")  # silence the logged traceback
        try:
            status, body, _ = call("GET", "/api/search?q=anything")
        finally:
            sys.stderr.close()
            sys.stderr = stderr
        check("unexpected errors -> 500 without leaking internals",
              status == 500 and err_code(body) == "internal" and "secret" not in json.dumps(body),
              f"{status} {body}")
    finally:
        service.search = real_search
T("internal_errors", t_internal_errors)


# ---------------------------------------------------------------- 5. crawl jobs
def wait_for_job(timeout: float = 30) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, body, _ = call("GET", "/api/crawl")
        job = body.get("job")
        if job and job["status"] != "running":
            return job
        time.sleep(0.2)
    raise TimeoutError("crawl job did not finish")


def t_crawl():
    status, body, _ = call("GET", "/api/crawl")
    check("crawl status before any job -> null", status == 200 and body.get("job") is None,
          f"{body}")
    for bad, label in (({"url": "ftp://example.com"}, "non-http scheme"),
                       ({"url": "not a url"}, "relative url"),
                       ({}, "missing url"),
                       ({"url": SITE_URL, "depth": 9}, "depth out of range"),
                       ({"url": SITE_URL, "max_pages": "lots"}, "non-integer max_pages")):
        status, body, _ = call("POST", "/api/crawl", bad)
        check(f"crawl rejects {label} -> 400", status == 400, f"{status} {body}")

    set_config(crawl_delay=0.05)
    status, body, _ = call("POST", "/api/crawl", {"url": SITE_URL + "/index.html", "depth": 1})
    check("crawl start -> 202 with a running job",
          status == 202 and body["job"]["status"] == "running", f"{status} {body}")
    job = wait_for_job()
    check("crawl job finishes with every page indexed",
          job["status"] == "done" and job["pages"] == 3 and job["chunks"] >= 3, str(job))
    check("crawl job keeps a readable log",
          any("Page Two" in line for line in job["log"]), str(job["log"]))
    status, body, _ = call("GET", "/api/search?q=vexillology")
    check("crawled content is searchable right away",
          status == 200 and body["results"] and body["results"][0]["title"] == "Page Two",
          f"{body}")

    set_config(crawl_delay=1.0)
    status, body, _ = call("POST", "/api/crawl", {"url": SITE_URL + "/index.html", "depth": 1})
    status2, body2, _ = call("POST", "/api/crawl", {"url": SITE_URL + "/index.html"})
    check("second crawl while one runs -> 409 busy",
          status == 202 and status2 == 409 and err_code(body2) == "busy", f"{status} {status2}")
    status, body, _ = call("DELETE", "/api/crawl")
    check("cancel a running crawl -> 200", status == 200, f"{status} {body}")
    job = wait_for_job()
    check("cancelled crawl stops early", job["status"] == "cancelled" and job["pages"] < 3,
          str(job))
    status, body, _ = call("DELETE", "/api/crawl")
    check("cancel with nothing running -> 409", status == 409 and err_code(body) == "not_running",
          f"{status}")

    status, body, _ = call("POST", "/api/crawl", {"url": "http://127.0.0.1:1/unreachable"})
    job = wait_for_job()
    check("crawl of an unreachable host finishes cleanly",
          job["status"] == "done" and job["pages"] == 0
          and any("error" in line for line in job["log"]), str(job))
    set_config()
T("crawl", t_crawl)


# ---------------------------------------------------------------- 6. concurrency
def t_concurrency():
    with ThreadPoolExecutor(max_workers=8) as pool:
        statuses = list(pool.map(lambda _: call("GET", "/api/search?q=fruit")[0], range(16)))
    check("16 concurrent searches all succeed", statuses == [200] * 16, str(statuses))
T("concurrency", t_concurrency)


# ---------------------------------------------------------------- summary
api.shutdown()
site.shutdown()
config.CONFIG_PATH = REAL_CONFIG_PATH
check("harness left the real config.json untouched",
      REAL_CONFIG_PATH.exists() == HAD_USER_CONFIG)
shutil.rmtree(SCRATCH, ignore_errors=True)

print("\n" + "=" * 62)
print(f"TOTAL: {len(PASS) + len(FAIL)}   PASS: {len(PASS)}   FAIL: {len(FAIL)}")
for name, detail in FAIL:
    print(f"  FAIL {name}" + (f"  | {detail[:300]}" if detail else ""))
sys.exit(1 if FAIL else 0)
