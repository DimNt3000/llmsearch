"""`llmsearch serve`: a JSON API over the index, for the mobile companion app.

Built on the standard library's http.server, so serving adds no
dependencies. Every endpoint except /api/health requires the bearer token
that `serve` prints on startup: on a shared network, the token is the
security boundary.
"""

from __future__ import annotations

import hmac
import json
import secrets
import socket
import sys
import threading
import time
import traceback
from contextlib import closing
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

from . import __version__, config, crawler, indexer, llm, service, websearch

MAX_BODY_BYTES = 64 * 1024
MAX_TEXT_CHARS = 1000
TOKEN_FILE = "server.token"

# The claude CLI backend runs one subprocess per call. Capping concurrent LLM
# work keeps a misbehaving client from forking an unbounded number of them.
_LLM_SLOTS = threading.BoundedSemaphore(2)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def load_or_create_token(rotate: bool = False) -> str:
    """Reuse the stored token so a paired phone survives server restarts."""
    path = config.data_dir() / TOKEN_FILE
    if not rotate and path.is_file():
        token = path.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(18)
    path.write_text(token + "\n", encoding="utf-8")
    return token


def lan_addresses() -> list[str]:
    """Best-effort list of this machine's LAN IPv4 addresses."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # a UDP connect sends no packets
            found.append(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if ip not in found and not ip.startswith("127."):
                found.append(ip)
    except OSError:
        pass
    return found


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


# --------------------------------------------------------------------------
# Background crawls
# --------------------------------------------------------------------------

class CrawlJobs:
    """One crawl at a time, observable and cancellable over the API."""

    LOG_KEEP = 200

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._job: dict | None = None

    def snapshot(self) -> dict | None:
        with self._lock:
            if self._job is None:
                return None
            job = dict(self._job)
            job["log"] = job["log"][-50:]
            return job

    def start(self, url: str, depth: int, max_pages: int, same_domain: bool,
              cfg: dict) -> dict | None:
        with self._lock:
            if self._job and self._job["status"] == "running":
                raise ApiError(409, "busy", "A crawl is already running.")
            self._stop.clear()
            job = self._job = {
                "id": secrets.token_hex(6), "url": url, "depth": depth,
                "max_pages": max_pages, "same_domain": same_domain,
                "status": "running", "pages": 0, "chunks": 0, "error": None,
                "log": [], "started_at": _now(), "finished_at": None,
            }
        threading.Thread(target=self._run, args=(job, cfg), daemon=True).start()
        return self.snapshot()

    def cancel(self) -> dict | None:
        with self._lock:
            if not self._job or self._job["status"] != "running":
                raise ApiError(409, "not_running", "No crawl is running.")
        self._stop.set()
        return self.snapshot()

    def _log(self, job: dict, line: str) -> None:
        with self._lock:
            job["log"].append(line)
            del job["log"][:-self.LOG_KEEP]

    def _run(self, job: dict, cfg: dict) -> None:
        def on_page(page: dict, n: int) -> None:
            with self._lock:
                job["pages"] += 1
                job["chunks"] += n
            self._log(job, f"+ {page['title'][:90]} ({n} chunks)")

        status, error = "done", None
        try:
            with closing(indexer.open_db(config.db_path())) as conn:
                service.crawl_site(
                    conn, job["url"], cfg, depth=job["depth"], max_pages=job["max_pages"],
                    same_domain=job["same_domain"], log=lambda s: self._log(job, s.strip()),
                    on_page=on_page, should_stop=self._stop.is_set,
                )
            if self._stop.is_set():
                status = "cancelled"
        except Exception as exc:  # noqa: BLE001 - reported to the client, not raised
            status, error = "error", str(exc) or type(exc).__name__
        with self._lock:
            job.update(status=status, error=error, finished_at=_now())


# --------------------------------------------------------------------------
# Request parsing and validation
# --------------------------------------------------------------------------

def _text(data: dict, key: str, max_len: int = MAX_TEXT_CHARS) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(400, "bad_request", f"'{key}' is required.")
    value = value.strip()
    if len(value) > max_len:
        raise ApiError(400, "bad_request", f"'{key}' is too long (max {max_len} characters).")
    return value


def _int(value: object, name: str, default: int, lo: int, hi: int) -> int:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ApiError(400, "bad_request", f"'{name}' must be an integer.")
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ApiError(400, "bad_request", f"'{name}' must be an integer.") from None
    if not lo <= number <= hi:
        raise ApiError(400, "bad_request", f"'{name}' must be between {lo} and {hi}.")
    return number


def _bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class Request:
    def __init__(self, handler: "Handler", query: dict[str, list[str]]) -> None:
        self.handler = handler
        self.server: ApiServer = handler.server  # type: ignore[assignment]
        self.query = query
        # Loaded per request, so `llmsearch config set` takes effect live.
        self.cfg = config.load_config()
        self._body: dict | None = None

    def param(self, name: str, required: bool = False) -> str | None:
        values = self.query.get(name)
        value = values[0].strip() if values else ""
        if not value:
            if required:
                raise ApiError(400, "bad_request", f"Query parameter '{name}' is required.")
            return None
        if len(value) > MAX_TEXT_CHARS:
            raise ApiError(400, "bad_request",
                           f"'{name}' is too long (max {MAX_TEXT_CHARS} characters).")
        return value

    def flag(self, name: str) -> bool:
        return _bool(self.param(name) or "")

    def json(self) -> dict:
        if self._body is None:
            self._body = self.handler.read_json()
        return self._body


def _open_index():
    return closing(indexer.open_db(config.db_path()))


def _result(r: dict) -> dict:
    return {
        "title": r.get("title") or r.get("url", ""),
        "url": r.get("url", ""),
        "score": round(float(r.get("score", 0.0)), 4),
        "snippet": r.get("snippet", ""),
        "text": r.get("text", ""),
    }


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

def h_health(req: Request):
    return 200, {"ok": True, "service": "llmsearch", "version": __version__}


def h_stats(req: Request):
    with _open_index() as conn:
        stats = indexer.stats(conn)
    backend = llm.resolve_backend(req.cfg)
    model = {"api": req.cfg["model"], "claude-cli": req.cfg["cli_model"]}.get(backend)
    return 200, {**stats, "backend": backend, "model": model,
                 "llm_available": backend != "off", "version": __version__}


def h_search(req: Request):
    query = req.param("q", required=True)
    smart, rerank = req.flag("smart"), req.flag("rerank")
    with _open_index() as conn:
        if smart or rerank:
            with _LLM_SLOTS:
                res = service.search(conn, query, req.cfg, smart=smart, rerank=rerank)
        else:
            res = service.search(conn, query, req.cfg)
    res["results"] = [_result(r) for r in res["results"]]
    return 200, res


def h_ask(req: Request):
    data = req.json()
    question = _text(data, "question")
    with _open_index() as conn, _LLM_SLOTS:
        return 200, service.ask(conn, question, req.cfg, smart=_bool(data.get("smart")))


def h_summarize(req: Request):
    query = _text(req.json(), "query")
    with _open_index() as conn, _LLM_SLOTS:
        return 200, service.summarize(conn, query, req.cfg)


def h_web(req: Request):
    query = req.param("q", required=True)
    n = _int(req.param("n"), "n", req.cfg["web_results"], 1, 20)
    return 200, service.web_search(query, req.cfg, n=n)


def h_web_ask(req: Request):
    query = _text(req.json(), "query")
    with _LLM_SLOTS:
        return 200, service.web_ask(query, req.cfg)


def h_crawl_status(req: Request):
    return 200, {"job": req.server.jobs.snapshot()}


def h_crawl_start(req: Request):
    data = req.json()
    url = _text(data, "url", max_len=2000)
    parts = urlparse(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ApiError(400, "bad_request", "'url' must be an absolute http(s) URL.")
    depth = _int(data.get("depth"), "depth", 1, 0, 3)
    max_pages = _int(data.get("max_pages"), "max_pages", 15, 1, 100)
    same_domain = not _bool(data.get("all_domains"))
    return 202, {"job": req.server.jobs.start(url, depth, max_pages, same_domain, req.cfg)}


def h_crawl_cancel(req: Request):
    return 200, {"job": req.server.jobs.cancel()}


Endpoint = Callable[[Request], "tuple[int, dict | None]"]

# (method, path) -> (handler, requires the bearer token)
ROUTES: dict[tuple[str, str], tuple[Endpoint, bool]] = {
    ("GET", "/api/health"): (h_health, False),
    ("GET", "/api/stats"): (h_stats, True),
    ("GET", "/api/search"): (h_search, True),
    ("POST", "/api/ask"): (h_ask, True),
    ("POST", "/api/summarize"): (h_summarize, True),
    ("GET", "/api/web"): (h_web, True),
    ("POST", "/api/web/ask"): (h_web_ask, True),
    ("GET", "/api/crawl"): (h_crawl_status, True),
    ("POST", "/api/crawl"): (h_crawl_start, True),
    ("DELETE", "/api/crawl"): (h_crawl_cancel, True),
}
_PATHS = {path for _, path in ROUTES}


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = f"llmsearch/{__version__}"
    sys_version = ""  # do not advertise the Python version

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # replaced by the one-line access log in _dispatch

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def do_OPTIONS(self) -> None:
        self._respond(204, None)  # CORS preflight

    def _dispatch(self, method: str) -> None:
        started = time.perf_counter()
        parts = urlparse(self.path)
        path = parts.path.rstrip("/") or "/"
        try:
            route = ROUTES.get((method, path))
            if route is None:
                if path in _PATHS:
                    raise ApiError(405, "method_not_allowed", f"{method} is not supported on {path}.")
                raise ApiError(404, "not_found", f"No endpoint at {path}.")
            endpoint, needs_token = route
            if needs_token and not self._authorized():
                raise ApiError(401, "unauthorized", "Missing or invalid token.")
            status, payload = endpoint(Request(self, parse_qs(parts.query)))
        except ApiError as exc:
            status, payload = exc.status, _error(exc.code, exc.message)
        except service.NoContext as exc:
            status, payload = 422, _error("no_context", str(exc))
        except llm.LLMUnavailable as exc:
            status, payload = 503, _error("llm_unavailable", str(exc))
        except llm.LLMError as exc:
            status, payload = 502, _error("llm_error", str(exc))
        except websearch.WebSearchError as exc:
            status, payload = 502, _error("web_error", str(exc))
        except crawler.CrawlError as exc:
            status, payload = 502, _error("fetch_error", str(exc))
        except Exception:  # noqa: BLE001 - never leak internals to the client
            traceback.print_exc()
            status, payload = 500, _error("internal", "Internal server error.")
        self._respond(status, payload)
        elapsed = (time.perf_counter() - started) * 1000
        self.server.access_log(f"{method:6} {path:16} {status}  {elapsed:7.0f} ms")  # type: ignore[attr-defined]

    def _authorized(self) -> bool:
        scheme, _, value = (self.headers.get("Authorization") or "").partition(" ")
        if scheme.lower() != "bearer":
            return False
        token: str = self.server.token  # type: ignore[attr-defined]
        return hmac.compare_digest(value.strip().encode(), token.encode())

    def read_json(self) -> dict:
        try:
            length = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            raise ApiError(400, "bad_request", "Invalid Content-Length.") from None
        if length > MAX_BODY_BYTES:
            self._drain(length)
            raise ApiError(413, "payload_too_large",
                           f"Request body exceeds {MAX_BODY_BYTES // 1024} KB.")
        raw = self.rfile.read(length) if length else b""
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ApiError(400, "bad_json", "Request body must be valid UTF-8 JSON.") from None
        if not isinstance(data, dict):
            raise ApiError(400, "bad_json", "Request body must be a JSON object.")
        return data

    def _drain(self, length: int) -> None:
        """Consume an oversized body (up to 1 MB) so the 413 reaches the client."""
        remaining = min(length, 1024 * 1024)
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def _respond(self, status: int, payload: dict | None) -> None:
        body = b"" if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            # The token, not the origin, is the access boundary. Allowing any
            # origin lets the app's web build reach the server during development.
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()
            if body:
                self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # the client went away, for example after cancelling a request


class ApiServer(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server silently bind the same
    # port; leaving it off there turns that into a clear "address in use".
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, address: tuple[str, int], token: str,
                 log: Callable[[str], None] = print) -> None:
        super().__init__(address, Handler)
        self.token = token
        self.jobs = CrawlJobs()
        self.access_log = lambda line: log(f"{time.strftime('%H:%M:%S')}  {line}")


def make_server(host: str, port: int, token: str,
                log: Callable[[str], None] = print) -> ApiServer:
    return ApiServer((host, port), token, log)
