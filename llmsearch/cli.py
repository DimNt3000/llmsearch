"""Command-line interface for llmsearch."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from . import config, crawler, indexer, llm, websearch

TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".text", ".html", ".htm"}


class C:
    """ANSI colors (auto-disabled when not a TTY or NO_COLOR is set)."""
    enabled = True
    BOLD, DIM, CYAN, YELLOW, GREEN, RED, RESET = (
        "\x1b[1m", "\x1b[2m", "\x1b[36m", "\x1b[33m", "\x1b[32m", "\x1b[31m", "\x1b[0m"
    )

    @classmethod
    def setup(cls, no_color: bool) -> None:
        cls.enabled = (
            not no_color and not os.environ.get("NO_COLOR")
            and sys.stdout.isatty()
        )
        if cls.enabled and sys.platform == "win32":
            os.system("")  # enable ANSI escape processing in the Windows console
        if not cls.enabled:
            cls.BOLD = cls.DIM = cls.CYAN = cls.YELLOW = cls.GREEN = cls.RED = cls.RESET = ""


def _print(text: str = "", end: str = "\n") -> None:
    try:
        print(text, end=end, flush=True)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "utf-8"
        print(text.encode(enc, errors="replace").decode(enc), end=end, flush=True)


# Crawled or model-generated text must not be able to inject terminal escape
# sequences. Remove well-formed CSI/OSC/Fe sequences first (so no inert
# "[31m" residue is shown), then sweep any remaining C0/C1 control bytes
# (except \t \n) as a backstop against malformed sequences.
_ANSI_SEQ_RE = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"              # ESC [ ... CSI
    r"|\x9b[0-?]*[ -/]*[@-~]"               # C1 CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # ESC ] ... OSC, BEL/ST-terminated
    r"|\x1b[@-_]"                           # other ESC Fe
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _clean(text: str) -> str:
    return _CONTROL_RE.sub("", _ANSI_SEQ_RE.sub("", text))


def _highlight(snippet: str, query: str) -> str:
    if not C.enabled:
        return snippet
    for term in set(indexer.tokenize(query)):
        snippet = re.sub(
            rf"(?i)\b({re.escape(term)})\b", f"{C.YELLOW}\\1{C.RESET}", snippet
        )
    return snippet


def _print_results(results: list[dict], query: str, cfg: dict,
                   as_json: bool = False) -> None:
    if as_json:
        payload = [
            {k: r.get(k) for k in ("title", "url", "score", "snippet")} for r in results
        ]
        _print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if not results:
        _print(f"{C.DIM}No results.{C.RESET}")
        return
    for i, r in enumerate(results, 1):
        snippet = _clean(r.get("snippet") or indexer.make_snippet(
            r.get("text", ""), query, cfg["snippet_chars"]
        ))
        score = f"{C.DIM}{r['score']:.3f}{C.RESET}" if "score" in r else ""
        _print(f"{C.BOLD}{i:2}. {C.CYAN}{_clean(r.get('title') or r['url'])}{C.RESET}  {score}")
        _print(f"    {C.DIM}{_clean(r['url'])}{C.RESET}")
        if snippet:
            _print(f"    {_highlight(snippet, query)}")
        _print()


def _print_source_list(sources: list[dict]) -> None:
    _print(f"\n{C.BOLD}Sources{C.RESET}")
    for i, s in enumerate(sources, 1):
        _print(f"  [{i}] {_clean(s.get('title', '?'))} "
               f"{C.DIM}- {_clean(s.get('url', '?'))}{C.RESET}")


def _stream_printer(chunk: str) -> None:
    _print(_clean(chunk), end="")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_crawl(args, cfg) -> int:
    conn = indexer.open_db(config.db_path())
    total_chunks = pages = 0
    _print(f"Crawling {C.CYAN}{args.url}{C.RESET} "
           f"(depth {args.depth}, max {args.max_pages} pages)...")
    for page in crawler.crawl(
        args.url, cfg, depth=args.depth, max_pages=args.max_pages,
        same_domain=not args.all_domains, log=lambda s: _print(f"{C.DIM}{_clean(s)}{C.RESET}"),
    ):
        n = indexer.add_document(conn, page["url"], page["title"], page["text"], "web", cfg)
        pages += 1
        total_chunks += n
        _print(f"  {C.GREEN}+{C.RESET} {_clean(page['title'])[:70]} {C.DIM}({n} chunks){C.RESET}")
    _print(f"\nIndexed {C.BOLD}{pages}{C.RESET} pages ({total_chunks} chunks).")
    return 0


def cmd_add(args, cfg) -> int:
    conn = indexer.open_db(config.db_path())
    files: list[Path] = []
    for raw in args.paths:
        p = Path(raw)
        if p.is_dir():
            files.extend(f for f in sorted(p.rglob("*"))
                         if f.is_file() and f.suffix.lower() in TEXT_EXTS)
        elif p.is_file():
            files.append(p)
        else:
            _print(f"{C.RED}not found:{C.RESET} {raw}")
    if not files:
        _print(f"{C.DIM}Nothing to index.{C.RESET}")
        return 1
    total_chunks = 0
    for f in files:
        raw_text = f.read_text(encoding="utf-8", errors="replace")
        if f.suffix.lower() in (".html", ".htm"):
            title, text, _ = crawler.extract(raw_text, f.as_uri())
            title = title or f.stem
        else:
            title, text = f.stem.replace("-", " ").replace("_", " "), raw_text
        n = indexer.add_document(conn, f.resolve().as_uri(), title, text, "file", cfg)
        total_chunks += n
        _print(f"  {C.GREEN}+{C.RESET} {f} {C.DIM}({n} chunks){C.RESET}")
    _print(f"\nIndexed {C.BOLD}{len(files)}{C.RESET} files ({total_chunks} chunks).")
    return 0


def cmd_search(args, cfg) -> int:
    conn = indexer.open_db(config.db_path())
    query = " ".join(args.query)
    queries = [query]
    use_rerank = args.rerank

    if args.smart:
        try:
            variants = llm.expand_query(query, cfg)
            if variants:
                _print(f"{C.DIM}Also searching: {'; '.join(variants)}{C.RESET}\n")
                queries += variants
        except llm.LLMUnavailable:
            _print(f"{C.DIM}LLM unavailable - falling back to plain BM25{C.RESET}")
            use_rerank = False  # rerank would hit the same missing backend

    rankings = [indexer.search_chunks(conn, q, cfg) for q in queries]
    chunks = rankings[0] if len(rankings) == 1 else indexer.rrf_merge(rankings)
    results = indexer.group_by_doc(chunks, cfg["top_k"] * 2 if use_rerank else cfg["top_k"])

    if use_rerank and results:
        try:
            results = llm.rerank(query, results, cfg, keep=cfg["top_k"])
        except llm.LLMUnavailable:
            _print(f"{C.DIM}LLM unavailable - falling back to plain BM25{C.RESET}")
            results = results[:cfg["top_k"]]

    _print_results(results, query, cfg, as_json=args.json)
    return 0


def cmd_ask(args, cfg) -> int:
    conn = indexer.open_db(config.db_path())
    question = " ".join(args.question)
    chunks = indexer.search_chunks(conn, question, cfg)
    if args.smart:
        variants = llm.expand_query(question, cfg)
        if variants:
            rankings = [chunks] + [indexer.search_chunks(conn, v, cfg) for v in variants]
            chunks = indexer.rrf_merge(rankings)
    contexts = indexer.top_contexts(chunks, cfg["ask_contexts"])
    if not contexts:
        _print(f"{C.DIM}The index has nothing relevant. "
               f"Use `crawl` or `add` to index content first.{C.RESET}")
        return 1
    llm.answer(question, contexts, cfg, stream_to=_stream_printer)
    _print()
    _print_source_list(contexts)
    return 0


def cmd_summarize(args, cfg) -> int:
    conn = indexer.open_db(config.db_path())
    query = " ".join(args.query)
    chunks = indexer.search_chunks(conn, query, cfg)
    results = indexer.group_by_doc(chunks, cfg["top_k"])
    if not results:
        _print(f"{C.DIM}No results to summarize.{C.RESET}")
        return 1
    llm.summarize(query, results, cfg, stream_to=_stream_printer)
    _print()
    _print_source_list(results)
    return 0


def cmd_web(args, cfg) -> int:
    query = " ".join(args.query)
    results = websearch.ddg_search(query, args.num or cfg["web_results"], cfg)
    if not args.ask:
        _print_results(results, query, cfg, as_json=args.json)
        return 0

    # --ask: fetch the top pages and answer from their content
    pages: list[dict] = []
    for r in results[:4]:
        try:
            _print(f"{C.DIM}fetching {_clean(r['url'])}...{C.RESET}")
            page = crawler.fetch_page_text(r["url"], cfg)
            if page["text"].strip():
                pages.append(page)
        except crawler.CrawlError as exc:
            _print(f"{C.DIM}  skipped: {_clean(str(exc))}{C.RESET}")
        if len(pages) >= 3:
            break
    if not pages:
        _print(f"{C.RED}Could not fetch any result pages.{C.RESET}")
        return 1
    _print()
    llm.answer(query, pages, cfg, stream_to=_stream_printer)
    _print()
    _print_source_list(pages)
    return 0


def cmd_stats(args, cfg) -> int:
    conn = indexer.open_db(config.db_path())
    s = indexer.stats(conn)
    _print(f"{C.BOLD}Index{C.RESET}  {config.db_path()}")
    _print(f"  documents:    {s['documents']}")
    _print(f"  chunks:       {s['chunks']}")
    _print(f"  unique terms: {s['unique_terms']}")
    for source, count in s["by_source"].items():
        _print(f"  {C.DIM}{source}: {count}{C.RESET}")
    backend = llm.resolve_backend(cfg)
    model = cfg["model"] if backend == "api" else cfg["cli_model"]
    label = f"{backend} ({model})" if backend != "off" else "off (no credentials found)"
    _print(f"{C.BOLD}LLM{C.RESET}    {label}")
    return 0


def cmd_config(args, cfg) -> int:
    if args.action == "set":
        cfg = config.set_value(cfg, args.key, args.value)
        config.save_config(cfg)
        _print(f"{args.key} = {cfg[args.key]}")
    elif args.action == "get":
        if args.key not in config.DEFAULTS:
            _print(f"{C.RED}unknown key:{C.RESET} {args.key}")
            return 1
        _print(f"{cfg[args.key]}")
    else:  # show
        for key in sorted(config.DEFAULTS):
            marker = "" if cfg[key] == config.DEFAULTS[key] else f" {C.YELLOW}*{C.RESET}"
            _print(f"  {key:18} = {cfg[key]}{marker}")
        _print(f"\n{C.DIM}* = changed from default. "
               f"Set with: config set KEY VALUE{C.RESET}")
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmsearch",
        description="A tiny personal search engine with Claude-powered smarts.",
        epilog=(
            "examples:\n"
            "  llmsearch crawl https://example.com --depth 1\n"
            "  llmsearch add ./samples\n"
            "  llmsearch search \"inverted index\"\n"
            "  llmsearch search \"ranking\" --smart --rerank\n"
            "  llmsearch ask \"how does BM25 length normalization work?\"\n"
            "  llmsearch web \"latest python release\" --ask\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-color", action="store_true", help="disable colored output")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("crawl", help="crawl a website into the index")
    p.add_argument("url")
    p.add_argument("--depth", type=int, default=1, help="link depth (default 1)")
    p.add_argument("--max-pages", type=int, default=30)
    p.add_argument("--all-domains", action="store_true",
                   help="follow links off the start domain")
    p.set_defaults(func=cmd_crawl)

    p = sub.add_parser("add", help="index local text/markdown/html files")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_add)

    p = sub.add_parser("search", help="BM25 search over the index")
    p.add_argument("query", nargs="+")
    p.add_argument("--smart", action="store_true",
                   help="LLM query expansion + rank fusion")
    p.add_argument("--rerank", action="store_true", help="LLM reranking of results")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("ask", help="answer a question from indexed content (RAG)")
    p.add_argument("question", nargs="+")
    p.add_argument("--smart", action="store_true",
                   help="LLM query expansion before retrieval")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("summarize", help="LLM digest of search results")
    p.add_argument("query", nargs="+")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("web", help="live DuckDuckGo search")
    p.add_argument("query", nargs="+")
    p.add_argument("-n", "--num", type=int, help="number of results")
    # --ask streams an LLM answer, which has no JSON form; reject the combo.
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--ask", action="store_true",
                      help="fetch top pages and answer from them")
    mode.add_argument("--json", action="store_true", help="machine-readable output")
    p.set_defaults(func=cmd_web)

    p = sub.add_parser("stats", help="index statistics")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("config", help="show or change settings")
    p.add_argument("action", nargs="?", default="show", choices=["show", "get", "set"])
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    p.set_defaults(func=cmd_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
        sys.stderr.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    C.setup(args.no_color)
    config.load_dotenv()
    cfg = config.load_config()
    if args.command == "config" and args.action in ("get", "set") and not args.key:
        _print(f"{C.RED}config {args.action} requires a KEY{C.RESET}")
        return 2
    if args.command == "config" and args.action == "set" and args.value is None:
        _print(f"{C.RED}config set requires KEY VALUE{C.RESET}")
        return 2
    try:
        return args.func(args, cfg)
    except KeyboardInterrupt:
        _print(f"\n{C.DIM}interrupted{C.RESET}")
        return 130
    except (llm.LLMError, websearch.WebSearchError, crawler.CrawlError,
            KeyError, ValueError) as exc:
        # str() of a KeyError is the repr of its message; unwrap it so the
        # user does not see stray quotes around the error text.
        msg = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        _print(f"{C.RED}error:{C.RESET} {msg}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
