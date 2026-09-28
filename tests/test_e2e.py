# -*- coding: utf-8 -*-
"""Offline end-to-end test harness for llmsearch.

Run it with:

    python tests/test_e2e.py

Exits 0 when every check passes, 1 otherwise, so it works as a CI gate.

Two properties are deliberate. It makes **no network calls and no LLM calls**,
which keeps it fast, free, and runnable anywhere. And it never writes inside
the repository: the index lives in a temporary directory pointed at by
LLMSEARCH_DATA_DIR, and the config path is monkeypatched, so your own index
and settings are left untouched.

Checks that need something not present on the machine (the `anthropic`
package, Python 3.11's tomllib) report as SKIP rather than failing.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
RUNPY = PROJECT / "run.py"
sys.path.insert(0, str(PROJECT))

import llmsearch.cli as cli  # noqa: E402
import llmsearch.config as config  # noqa: E402
import llmsearch.indexer as indexer  # noqa: E402
import llmsearch.llm as llm  # noqa: E402
from llmsearch.cli import _clean  # noqa: E402
from llmsearch.indexer import chunk_text, make_snippet, rrf_merge, tokenize  # noqa: E402

PASS: list[tuple[str, str]] = []
FAIL: list[tuple[str, str]] = []
SKIP: list[tuple[str, str]] = []

SCRATCH = Path(tempfile.mkdtemp(prefix="llmsearch-tests-"))
# Captured before any test monkeypatches config.CONFIG_PATH, so the final
# assertion can tell whether the real settings file was disturbed.
REAL_CONFIG_PATH = config.CONFIG_PATH
HAD_USER_CONFIG = REAL_CONFIG_PATH.exists()


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append((name, detail))
    print(("PASS " if cond else "FAIL ") + name + ("  | " + detail if detail and not cond else ""))


def skip(name: str, why: str) -> None:
    SKIP.append((name, why))
    print(f"SKIP {name}  | {why}")


def run_cli(args: list[str], data_dir: Path, timeout: int = 60):
    """Invoke the CLI as a subprocess against an isolated index."""
    env = {**os.environ, "LLMSEARCH_DATA_DIR": str(data_dir), "NO_COLOR": "1"}
    return subprocess.run([sys.executable, str(RUNPY)] + args, capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, env=env)


def inproc(args: list[str], cfg_path: Path, data_dir: Path):
    """Run cli.main in-process with a monkeypatched config path.

    Returns (exit_code, stdout, uncaught_exception). An exception here means
    the CLI leaked a traceback to the user, which is itself a failure.
    """
    config.CONFIG_PATH = Path(cfg_path)
    os.environ["LLMSEARCH_DATA_DIR"] = str(data_dir)
    buf, exc = io.StringIO(), None
    try:
        with contextlib.redirect_stdout(buf):
            rc = cli.main(list(args))
    except SystemExit as e:
        rc = e.code
    except Exception as e:  # noqa: BLE001 - the point is to catch everything
        rc, exc = None, e
    return rc, buf.getvalue(), exc


def T(name: str, fn) -> None:
    try:
        fn()
    except Exception:
        FAIL.append((name, "test raised"))
        print(f"FAIL {name}  | test raised:\n{traceback.format_exc(limit=3)}")


# ---------------------------------------------------------------- fixtures
for sub in ("db1", "db2", "empty", "docs"):
    (SCRATCH / sub).mkdir(parents=True, exist_ok=True)
DOCS = SCRATCH / "docs"
(DOCS / "apples.md").write_text(
    "# Apples\n\nApples grow on trees in orchards. The zylophant fruit is rare.\n\n"
    "Apple cultivation requires patience and careful pruning every season.", encoding="utf-8")
(DOCS / "bananas.txt").write_text(
    "Bananas are yellow tropical fruit. Banana plants are technically herbs.\n\n"
    "Banana exports drive several economies.", encoding="utf-8")
(DOCS / "greek.md").write_text(
    "# Ελληνικά\n\nΗ μηχανή αναζήτησης υποστηρίζει ελληνικό κείμενο χωρίς πρόβλημα.",
    encoding="utf-8")
(DOCS / "page.html").write_text(
    "<html><head><title>Fancy Title</title><style>body{color:red}</style>"
    "<script>alert('nope')</script></head><body><p>Visible html body text about "
    "kumquats.</p></body></html>", encoding="utf-8")
(DOCS / "empty.txt").write_text("", encoding="utf-8")
(DOCS / "ignore.py").write_text("print('should not be indexed')", encoding="utf-8")

if HAD_USER_CONFIG:
    print("note: a config.json exists; subprocess checks run with your settings\n")


# ---------------------------------------------------------------- 1. indexing
def t_add():
    r = run_cli(["add", str(DOCS)], SCRATCH / "db1")
    check("add: directory tree rc=0", r.returncode == 0, r.stdout + r.stderr)
    check("add: unsupported extensions ignored", "ignore.py" not in r.stdout, r.stdout)
    check("add: 5 files indexed", "5 files" in r.stdout, r.stdout)
T("add", t_add)


def t_readd():
    db = SCRATCH / "db1" / "index.db"
    before = sqlite3.connect(db).execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    run_cli(["add", str(DOCS / "apples.md")], SCRATCH / "db1")
    conn = sqlite3.connect(db)
    after = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    orphans = conn.execute(
        "SELECT COUNT(*) FROM postings WHERE chunk_id NOT IN (SELECT id FROM chunks)"
    ).fetchone()[0]
    check("re-add: replaces instead of duplicating", before == after, f"{before}->{after}")
    check("re-add: no orphan postings left behind", orphans == 0, str(orphans))
T("readd", t_readd)


# ---------------------------------------------------------------- 2. search
def t_search():
    r = run_cli(["search", "zylophant"], SCRATCH / "db1")
    check("search: rare term hits the right doc", "apples" in r.stdout.lower(), r.stdout)
    r2 = run_cli(["search", "banana", "economies"], SCRATCH / "db1")
    check("search: multi-word query", "bananas" in r2.stdout.lower(), r2.stdout)
    r3 = run_cli(["search", "qqqqzzzz"], SCRATCH / "db1")
    check("search: no results reported cleanly",
          "No results" in r3.stdout and r3.returncode == 0, r3.stdout)
    r4 = run_cli(["search", "the", "of", "and"], SCRATCH / "db1")
    check("search: stopword-only query does not crash",
          r4.returncode == 0 and "Traceback" not in r4.stderr, r4.stderr)
    r5 = run_cli(["search", "αναζήτησης"], SCRATCH / "db1")
    check("search: non-ASCII query", "greek" in r5.stdout.lower(), r5.stdout)
    r6 = run_cli(["search", "kumquats"], SCRATCH / "db1")
    check("add: HTML stripped, title extracted",
          "Fancy Title" in r6.stdout and "alert(" not in r6.stdout, r6.stdout)
    r7 = run_cli(["search", "apples", "--json"], SCRATCH / "db1")
    try:
        payload = json.loads(r7.stdout)
        ok = isinstance(payload, list) and all(
            set(p) == {"title", "url", "score", "snippet"} for p in payload)
    except Exception as e:  # noqa: BLE001
        ok, payload = False, str(e)
    check("search: --json emits valid JSON", ok, str(payload)[:200])
    r8 = run_cli(["search", "apples"], SCRATCH / "db1")
    check("search: NO_COLOR suppresses ANSI", "\x1b[" not in r8.stdout, repr(r8.stdout[:100]))
T("search", t_search)


def t_empty_index():
    r = run_cli(["search", "anything"], SCRATCH / "empty")
    check("empty index: search is clean",
          r.returncode == 0 and "Traceback" not in r.stderr, r.stderr)
    r2 = run_cli(["stats"], SCRATCH / "empty")
    check("empty index: stats reports zeros", "documents:    0" in r2.stdout, r2.stdout)
    t0 = time.time()
    r3 = run_cli(["ask", "anything at all"], SCRATCH / "empty", timeout=30)
    dt = time.time() - t0
    check("empty index: ask exits early without calling the LLM",
          r3.returncode == 1 and dt < 10 and "nothing relevant" in r3.stdout,
          f"rc={r3.returncode} {dt:.1f}s")
T("empty_index", t_empty_index)


def t_stats():
    r = run_cli(["stats"], SCRATCH / "db1")
    conn = sqlite3.connect(SCRATCH / "db1" / "index.db")
    d = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    c = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    check("stats: counts match the database",
          f"documents:    {d}" in r.stdout and f"chunks:       {c}" in r.stdout, r.stdout)
    backend = llm.resolve_backend(dict(config.DEFAULTS))
    check("stats: reports the resolved backend", backend in r.stdout, f"{backend} / {r.stdout}")
T("stats", t_stats)


# ---------------------------------------------------------------- 3. CLI contract
def t_exitcodes():
    check("no args -> exit 2", run_cli([], SCRATCH / "db2").returncode == 2)
    check("unknown command -> exit 2", run_cli(["frobnicate"], SCRATCH / "db2").returncode == 2)
    check("search without query -> exit 2", run_cli(["search"], SCRATCH / "db2").returncode == 2)
    check("--help -> exit 0", run_cli(["--help"], SCRATCH / "db2").returncode == 0)
    for sub in ("crawl", "add", "search", "ask", "summarize", "web", "serve", "stats", "config"):
        check(f"{sub} --help -> exit 0", run_cli([sub, "--help"], SCRATCH / "db2").returncode == 0)
    r = run_cli(["add", str(SCRATCH / "nope-missing")], SCRATCH / "db2")
    check("add on a missing path -> clean exit 1",
          r.returncode == 1 and "Traceback" not in r.stderr, r.stderr)
    r2 = run_cli(["config", "get", "nosuchkey"], SCRATCH / "db2")
    check("config get unknown key -> clean exit 1",
          r2.returncode == 1 and "unknown key" in r2.stdout, r2.stdout)
    r3 = run_cli(["config", "get"], SCRATCH / "db2")
    check("config get without a key -> exit 2", r3.returncode == 2, str(r3.returncode))
    r4 = run_cli(["config", "show"], SCRATCH / "db2")
    check("config show lists every default",
          all(k in r4.stdout for k in config.DEFAULTS), r4.stdout)
    r5 = run_cli(["web", "x", "--ask", "--json"], SCRATCH / "db2")
    check("web --ask --json rejected by the parser", r5.returncode == 2, str(r5.returncode))
T("exitcodes", t_exitcodes)


def t_config_set():
    cfgp = SCRATCH / "db2" / "config.json"
    rc, out, exc = inproc(["config", "set", "top_k", "3"], cfgp, SCRATCH / "db1")
    check("config set coerces ints", rc == 0 and json.loads(cfgp.read_text())["top_k"] == 3, out)
    rc, out, exc = inproc(["config", "set", "bm25_b", "0.5"], cfgp, SCRATCH / "db1")
    check("config set coerces floats",
          rc == 0 and json.loads(cfgp.read_text())["bm25_b"] == 0.5, out)
    rc, out, exc = inproc(["config", "set", "nosuchkey", "5"], cfgp, SCRATCH / "db1")
    check("config set unknown key -> clean error", rc == 1 and exc is None,
          f"rc={rc} exc={exc!r} out={out}")
    # Regression: a non-numeric value used to escape as a raw ValueError traceback.
    rc, out, exc = inproc(["config", "set", "top_k", "abc"], cfgp, SCRATCH / "db1")
    check("config set with a bad value -> message, not traceback", exc is None,
          f"uncaught {type(exc).__name__}: {exc}")
    rc, out, exc = inproc(["search", "zylophant"], cfgp, SCRATCH / "db1")
    check("stored config is honored on the next run", rc == 0 and exc is None, f"{exc!r}")
T("config_set", t_config_set)


# ---------------------------------------------------------------- 4. LLM layer
def t_backend_off():
    cfgp = SCRATCH / "db2" / "config_off.json"
    cfgp.write_text('{"backend": "off"}', encoding="utf-8")
    rc, out, exc = inproc(["search", "zylophant"], cfgp, SCRATCH / "db1")
    check("backend off: plain search still works",
          rc == 0 and exc is None and "apples" in out.lower(), out[:200])
    rc, out, exc = inproc(["ask", "zylophant"], cfgp, SCRATCH / "db1")
    check("backend off: ask explains how to enable one",
          rc == 1 and exc is None and "No LLM backend" in out, f"rc={rc} {out[:200]}")
    # Regression: --smart used to abort the whole search when no backend existed.
    rc, out, exc = inproc(["search", "zylophant", "--smart"], cfgp, SCRATCH / "db1")
    check("backend off: --smart degrades to plain BM25", rc == 0, f"rc={rc} out={out[:150]}")
T("backend_off", t_backend_off)


def t_backend_matrix():
    base = dict(config.DEFAULTS)
    for val in ("off", "api", "claude-cli"):
        check(f"resolve_backend honors explicit {val}",
              llm.resolve_backend({**base, "backend": val}) == val)
    auto = llm.resolve_backend({**base, "backend": "auto"})
    check("resolve_backend auto returns a known backend",
          auto in ("api", "claude-cli", "off"), str(auto))
    usable = llm._api_importable() or bool(shutil.which("claude"))
    check("resolve_backend auto reports off only when nothing is usable",
          (auto != "off") == usable, f"auto={auto} usable={usable}")
    if llm._api_importable():
        code = ("import sys, os; sys.path.insert(0, r'%s'); "
                "os.environ['ANTHROPIC_API_KEY']='sk-test-fake'; "
                "import llmsearch.llm as llm, llmsearch.config as c; "
                "print(llm.resolve_backend({**c.DEFAULTS, 'backend': 'auto'}))" % PROJECT)
        # A cold `import anthropic` takes a few seconds, but it passed 30 on a
        # machine busy with a native build; the check is about behavior, not speed.
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, timeout=120)
        check("resolve_backend auto prefers the API when a key exists",
              r.stdout.strip() == "api", r.stdout + r.stderr)
    else:
        skip("resolve_backend auto prefers the API when a key exists",
             "anthropic package not installed")
T("backend_matrix", t_backend_matrix)


def t_fail_open():
    """Model output is untrusted input: garbage must never break a command."""
    real = llm.complete
    try:
        llm.complete = lambda *a, **k: "utter garbage, no json here"
        check("expand_query fails open to no variants",
              llm.expand_query("q", dict(config.DEFAULTS)) == [])
        results = [{"title": "a", "text": "x"}, {"title": "b", "text": "y"}]
        check("rerank fails open to the BM25 order",
              llm.rerank("q", results, dict(config.DEFAULTS), keep=2) == results)
        llm.complete = lambda *a, **k: 'Sure! ```json\n["alpha beta", "gamma"]\n``` hope that helps'
        check("expand_query parses JSON out of code fences",
              llm.expand_query("q", dict(config.DEFAULTS)) == ["alpha beta", "gamma"])
        llm.complete = lambda *a, **k: "The best order is [1, 0] obviously."
        check("rerank parses JSON out of prose",
              llm.rerank("q", results, dict(config.DEFAULTS), keep=2) == [results[1], results[0]])
    finally:
        llm.complete = real
T("fail_open", t_fail_open)


def t_cli_backend_errors():
    """The claude CLI backend must fail fast and say why."""
    import socket
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    dead_port = closed.getsockname()[1]
    closed.close()  # nothing listens on this port any more
    listening = socket.socket()
    listening.bind(("127.0.0.1", 0))
    listening.listen(1)
    live_port = listening.getsockname()[1]
    try:
        check("proxy check: no override is fine", llm._dead_local_proxy(None) is False)
        check("proxy check: remote URLs are never probed",
              llm._dead_local_proxy("https://api.anthropic.com") is False)
        check("proxy check: a stopped local proxy is detected",
              llm._dead_local_proxy(f"http://127.0.0.1:{dead_port}") is True)
        check("proxy check: a running local proxy is accepted",
              llm._dead_local_proxy(f"http://127.0.0.1:{live_port}") is False)
    finally:
        listening.close()

    real = (llm.shutil.which, llm.subprocess.run, llm._claude_base_url)
    calls: list[int] = []
    try:
        llm.shutil.which = lambda name: "claude"  # pretend the CLI is installed
        llm._claude_base_url = lambda: f"http://127.0.0.1:{dead_port}"
        llm.subprocess.run = lambda *a, **k: calls.append(1)
        t0 = time.time()
        try:
            llm._cli_complete("sys", "hi", dict(config.DEFAULTS), None)
            outcome = "no error"
        except llm.LLMUnavailable as exc:
            outcome = str(exc)
        # Regression: a stopped proxy used to cost ~200 s of CLI retries.
        check("dead proxy -> LLMUnavailable in seconds, CLI never started",
              "nothing is listening" in outcome and not calls and time.time() - t0 < 5,
              f"{outcome} calls={calls}")

        llm._claude_base_url = lambda: None
        llm.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(
            a, 1, stdout="API Error: Connection error.", stderr="")
        try:
            llm._cli_complete("sys", "hi", dict(config.DEFAULTS), None)
            outcome = "no error"
        except llm.LLMUnavailable as exc:
            outcome = f"misread as a login problem: {exc}"
        except llm.LLMError as exc:
            outcome = str(exc)
        # Regression: this used to report "unknown error" and drop the real message.
        check("CLI failure surfaces the error it printed on stdout",
              "Connection error" in outcome and "misread" not in outcome, outcome)

        expired = ('Failed to authenticate. API Error: 401 {"type":"error","error":'
                   '{"type":"authentication_error","message":"OAuth access token has '
                   'expired. Re-authenticate to continue."}}')
        envs: list[dict] = []

        def fake_run(*a, **k):
            envs.append(k.get("env") or {})
            return subprocess.CompletedProcess(a, 1, stdout=expired, stderr="")

        llm.subprocess.run = fake_run
        try:
            llm._cli_complete("sys", "hi", dict(config.DEFAULTS), None)
            outcome = "no error"
        except llm.LLMUnavailable as exc:
            outcome = str(exc)
        except llm.LLMError as exc:
            outcome = f"not reported as unavailable: {exc}"
        # Regression: an expired CLI login came back as a raw 401, after minutes of retries.
        check("expired CLI login -> LLMUnavailable that names the fix",
              "claude auth login" in outcome, outcome)
        check("the CLI runs with a small retry budget by default",
              bool(envs) and envs[-1].get("CLAUDE_CODE_MAX_RETRIES") == "2",
              str([e.get("CLAUDE_CODE_MAX_RETRIES") for e in envs]))
        try:
            llm._cli_complete("sys", "hi", {**config.DEFAULTS, "cli_max_retries": 5}, None)
        except llm.LLMError:
            pass
        check("cli_max_retries sets the CLI's retry budget",
              envs[-1].get("CLAUDE_CODE_MAX_RETRIES") == "5",
              str(envs[-1].get("CLAUDE_CODE_MAX_RETRIES")))
    finally:
        llm.shutil.which, llm.subprocess.run, llm._claude_base_url = real
T("cli_backend_errors", t_cli_backend_errors)


# ---------------------------------------------------------------- 5. retrieval internals
def t_chunking():
    text = "word " * 2000  # a single 10,000-char paragraph, no blank lines
    chunks = chunk_text(text.strip(), 1400, 200)
    check("chunk: oversized paragraph is split", len(chunks) >= 7, str(len(chunks)))
    check("chunk: sizes stay within the limit", all(len(c) <= 1400 for c in chunks))
    check("chunk: no content lost mid-document",
          any(text[5000:5040].strip() in c for c in chunks))
    check("chunk: empty input -> no chunks", chunk_text("", 1400, 200) == [])
    check("chunk: whitespace-only input -> no chunks", chunk_text("  \n\n  ", 1400, 200) == [])
    # Regression: overlap >= size made the hard-split stride zero and hung forever.
    code = ("import sys; sys.path.insert(0, r'%s'); from llmsearch.indexer import chunk_text; "
            "chunk_text('x' * 500, 100, 100); print('done')" % PROJECT)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
        hung = "done" not in r.stdout
    except subprocess.TimeoutExpired:
        hung = True
    check("chunk: overlap >= size cannot stall the loop", not hung,
          "chunk_text hangs when chunk_overlap >= chunk_chars")

    check("tokenize: underscores split tokens", tokenize("_foo_bar_") == ["foo", "bar"])
    check("tokenize: absurdly long tokens dropped", tokenize("a" * 41) == [])
    check("tokenize: Greek kept and lowercased", tokenize("Αναζήτηση") == ["αναζήτηση"])
    check("tokenize: stopwords dropped", tokenize("the of and") == [])
    check("tokenize: single characters dropped", tokenize("b parameter") == ["parameter"])

    long_text = ("filler words here " * 300) + " the zylophant appears here once " \
                + ("more filler " * 300)
    snip = make_snippet(long_text, "zylophant", 240)
    check("snippet: window contains the query term", "zylophant" in snip, snip[:120])
    check("snippet: respects the character budget", len(snip) <= 260, str(len(snip)))

    r1 = [{"chunk_id": 1, "score": 5}, {"chunk_id": 2, "score": 4}]
    r2 = [{"chunk_id": 2, "score": 9}, {"chunk_id": 3, "score": 1}]
    merged = rrf_merge([r1, r2])
    check("rrf: an item ranked by both lists wins",
          merged[0]["chunk_id"] == 2, str([m["chunk_id"] for m in merged]))
T("chunking", t_chunking)


def t_bm25():
    cfg = dict(config.DEFAULTS)
    conn = indexer.open_db(SCRATCH / "db2" / "bm25.db")
    indexer.add_document(conn, "u1", "d1", "gamma common words appear here", "file", cfg)
    indexer.add_document(conn, "u2", "d2", "gamma common words again present", "file", cfg)
    indexer.add_document(conn, "u3", "d3", "gamma plus the rareterm lives here", "file", cfg)
    rare = indexer.search_chunks(conn, "rareterm", cfg)
    common = indexer.search_chunks(conn, "gamma", cfg)
    check("bm25: rare term matches only its document",
          len(rare) == 1 and rare[0]["url"] == "u3")
    check("bm25: IDF ranks a rare term above a common one",
          rare[0]["score"] > max(r["score"] for r in common),
          f"{rare[0]['score']:.3f} vs {max(r['score'] for r in common):.3f}")

    conn2 = indexer.open_db(SCRATCH / "db2" / "len.db")
    indexer.add_document(conn2, "s", "short", "needle in a tiny doc", "file", cfg)
    indexer.add_document(conn2, "l", "long",
                         "needle " + ("padding words repeated endlessly " * 40), "file", cfg)
    res = indexer.search_chunks(conn2, "needle", cfg)
    check("bm25: length normalization favors the shorter document", res[0]["url"] == "s",
          str([(r["url"], round(r["score"], 3)) for r in res]))
    indexer.add_document(conn2, "e", "empty", "", "file", cfg)
    s = indexer.stats(conn2)
    check("empty document indexes without crashing",
          s["documents"] == 3 and s["chunks"] >= 2, str(s))
T("bm25", t_bm25)


# ---------------------------------------------------------------- 6. hostile input
def t_escape_injection():
    """A crawled page controls its own title and body; it must not control the terminal."""
    check("_clean removes whole CSI/OSC/C1 sequences",
          _clean("a\x1b[31mred\x1b]0;t\x07b\x9bXc\x00d\x7f") == "aredbcd",
          repr(_clean("a\x1b[31mred\x1b]0;t\x07b\x9bXc\x00d\x7f")))
    check("_clean removes an unterminated OSC",
          _clean("x\x1b]0;evil-title") == "x", repr(_clean("x\x1b]0;evil-title")))
    check("_clean keeps newlines, tabs, and non-ASCII text", _clean("α\nβ\tγ") == "α\nβ\tγ")

    evil = SCRATCH / "docs-evil"
    evil.mkdir(exist_ok=True)
    (evil / "evil.html").write_text(
        "<html><head><title>Evil \x1b]0;HACKED\x07 \x1b[2J title</title></head>"
        "<body><p>zzinjecttest content \x1b[31mFAKE-ERROR\x1b[0m and \x9b31m more</p>"
        "</body></html>", encoding="utf-8")
    run_cli(["add", str(evil / "evil.html")], SCRATCH / "db2")
    r = run_cli(["search", "zzinjecttest"], SCRATCH / "db2")
    check("crawled escapes never reach stdout",
          "\x1b" not in r.stdout and "\x9b" not in r.stdout, repr(r.stdout[:200]))
    check("sanitization keeps the visible words",
          "FAKE-ERROR" in r.stdout and "Evil" in r.stdout, repr(r.stdout[:200]))
T("escape_injection", t_escape_injection)


def t_dotenv():
    """A malformed .env must never take a command down with it."""
    envp = SCRATCH / "db2" / ".env"
    real_env_path = config.ENV_PATH
    keys = ("TESTKEY_PLAIN", "QUOTED", "TESTKEY_BOM", "TESTKEY_U16", "﻿TESTKEY_BOM")
    try:
        config.ENV_PATH = envp
        for k in keys:
            os.environ.pop(k, None)
        envp.write_text("TESTKEY_PLAIN=hello\n# comment\nQUOTED='v1'\n", encoding="utf-8")
        config.load_dotenv()
        check(".env: plain UTF-8 loads", os.environ.get("TESTKEY_PLAIN") == "hello")
        check(".env: surrounding quotes stripped", os.environ.get("QUOTED") == "v1")
        envp.write_text("TESTKEY_BOM=1\n", encoding="utf-8-sig")
        config.load_dotenv()
        check(".env: UTF-8 BOM does not corrupt the first key",
              os.environ.get("TESTKEY_BOM") == "1", "BOM stuck to the key name")
        # Regression: PowerShell's `>` writes UTF-16, which used to raise
        # "embedded null character" before any command could run.
        envp.write_bytes("TESTKEY_U16=1\n".encode("utf-16"))
        os.environ.pop("TESTKEY_U16", None)
        config.load_dotenv()
        check(".env: UTF-16 is decoded instead of crashing",
              os.environ.get("TESTKEY_U16") == "1", "utf-16 .env parsed to garbage")
    finally:
        config.ENV_PATH = real_env_path
        for k in keys:
            os.environ.pop(k, None)
T("dotenv", t_dotenv)


# ---------------------------------------------------------------- 7. packaging
def t_packaging():
    import llmsearch
    try:
        import tomllib
    except ImportError:  # tomllib landed in 3.11
        skip("pyproject: entry point and version", "tomllib requires Python 3.11+")
        tomllib = None
    if tomllib is not None:
        py = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
        check("pyproject: console script points at a real callable",
              py["project"]["scripts"]["llmsearch"] == "llmsearch.cli:main" and callable(cli.main))
        check("pyproject: version matches the package",
              py["project"]["version"] == llmsearch.__version__,
              f"{py['project']['version']} vs {llmsearch.__version__}")
    gi = (PROJECT / ".gitignore").read_text(encoding="utf-8")
    check(".gitignore covers the index, .env, and config",
          all(x in gi for x in ("data/", ".env", "config.json", "__pycache__")))
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    missing = [k for k in config.DEFAULTS if k not in readme]
    check("README documents every config key", not missing, str(missing))
T("packaging", t_packaging)


# ---------------------------------------------------------------- 8. concurrency
def t_concurrent():
    env = {**os.environ, "LLMSEARCH_DATA_DIR": str(SCRATCH / "db1"), "NO_COLOR": "1"}
    procs = [subprocess.Popen([sys.executable, str(RUNPY), "search", "apples"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
             for _ in range(3)]
    outs = [p.communicate(timeout=60) for p in procs]
    check("concurrent readers do not lock each other out",
          all(p.returncode == 0 for p in procs) and not any(b"locked" in e for _, e in outs),
          str([p.returncode for p in procs]))
T("concurrent", t_concurrent)


# ---------------------------------------------------------------- summary
check("harness left the real config.json untouched",
      REAL_CONFIG_PATH.exists() == HAD_USER_CONFIG)
shutil.rmtree(SCRATCH, ignore_errors=True)

print("\n" + "=" * 62)
total = len(PASS) + len(FAIL)
print(f"TOTAL: {total}   PASS: {len(PASS)}   FAIL: {len(FAIL)}   SKIP: {len(SKIP)}")
for name, detail in FAIL:
    print(f"  FAIL {name}" + (f"  | {detail[:300]}" if detail else ""))
sys.exit(1 if FAIL else 0)
