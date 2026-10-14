"""Search operations shared by the CLI and the HTTP server.

Every function here returns plain data and never prints, so `llmsearch`
on the command line and `llmsearch serve` for the mobile app run exactly
the same pipeline and only differ in how they present the result.
"""

from __future__ import annotations

import sqlite3
from typing import Callable

from . import crawler, indexer, llm, websearch

Log = Callable[[str], None]

FALLBACK_NOTICE = "LLM unavailable - falling back to plain BM25"


class NoContext(Exception):
    """Nothing was found to answer from; the message says what to do next."""


def _sources(items: list[dict], text_limit: int | None = None) -> list[dict]:
    return [
        {
            "n": i,
            "title": item.get("title") or item.get("url", ""),
            "url": item.get("url", ""),
            "text": (item.get("text") or "")[:text_limit] if text_limit else item.get("text", ""),
        }
        for i, item in enumerate(items, 1)
    ]


def search(conn: sqlite3.Connection, query: str, cfg: dict,
           smart: bool = False, rerank: bool = False) -> dict:
    """BM25 search, optionally widened by LLM query expansion and reranked.

    LLM features degrade to plain BM25 when no backend is available; the
    returned `notice` says so. Other LLM errors propagate to the caller.
    """
    queries = [query]
    variants: list[str] = []
    notice = None
    use_rerank = rerank

    if smart:
        try:
            variants = llm.expand_query(query, cfg)
            queries += variants
        except llm.LLMUnavailable:
            notice = FALLBACK_NOTICE
            use_rerank = False  # rerank would hit the same missing backend

    rankings = [indexer.search_chunks(conn, q, cfg) for q in queries]
    chunks = rankings[0] if len(rankings) == 1 else indexer.rrf_merge(rankings)
    top_k = cfg["top_k"]
    results = indexer.group_by_doc(chunks, top_k * 2 if use_rerank else top_k)

    if use_rerank and results:
        try:
            results = llm.rerank(query, results, cfg, keep=top_k)
        except llm.LLMUnavailable:
            notice = FALLBACK_NOTICE
            results = results[:top_k]

    for r in results:
        r["snippet"] = indexer.make_snippet(r.get("text", ""), query, cfg["snippet_chars"])
    return {"query": query, "variants": variants, "notice": notice, "results": results}


def retrieve_contexts(conn: sqlite3.Connection, question: str, cfg: dict,
                      smart: bool = False) -> list[dict]:
    chunks = indexer.search_chunks(conn, question, cfg)
    if smart:
        variants = llm.expand_query(question, cfg)
        if variants:
            rankings = [chunks] + [indexer.search_chunks(conn, v, cfg) for v in variants]
            chunks = indexer.rrf_merge(rankings)
    return indexer.top_contexts(chunks, cfg["ask_contexts"])


def ask(conn: sqlite3.Connection, question: str, cfg: dict, smart: bool = False,
        stream_to: llm.StreamFn | None = None) -> dict:
    """Retrieval-augmented answer with numbered sources matching its [n] citations."""
    contexts = retrieve_contexts(conn, question, cfg, smart=smart)
    if not contexts:
        raise NoContext("The index has nothing relevant. "
                        "Use `crawl` or `add` to index content first.")
    answer = llm.answer(question, contexts, cfg, stream_to=stream_to)
    return {"question": question, "answer": answer, "sources": _sources(contexts)}


def summarize(conn: sqlite3.Connection, query: str, cfg: dict,
              stream_to: llm.StreamFn | None = None) -> dict:
    chunks = indexer.search_chunks(conn, query, cfg)
    results = indexer.group_by_doc(chunks, cfg["top_k"])
    if not results:
        raise NoContext("No results to summarize.")
    summary = llm.summarize(query, results, cfg, stream_to=stream_to)
    return {"query": query, "summary": summary, "sources": _sources(results)}


def web_search(query: str, cfg: dict, n: int | None = None) -> dict:
    results = websearch.ddg_search(query, n or cfg["web_results"], cfg)
    return {"query": query, "results": results}


WEB_CONTEXT_CHARS = 5400  # page text sent to the model per web answer


def web_source_limit(pages: list[dict]) -> int:
    """Characters of each page the model sees: a fixed budget shared by the
    pages, so one page read directly is seen in full rather than cut short."""
    return max(1800, WEB_CONTEXT_CHARS // max(1, len(pages)))


def fetch_web_pages(query: str, cfg: dict, n: int | None = None,
                    log: Log | None = None) -> tuple[list[dict], list[str]]:
    """Fetch up to three readable pages to answer from.

    Sites named in the question are read directly, live; a search would
    rarely return a news front page as it is right now. Only when none of
    them can be read does this search the web and fetch the top results.
    """
    pages: list[dict] = []
    skipped: list[str] = []
    for url in websearch.direct_urls(query):
        try:
            if log:
                log(f"reading {url}...")
            page = crawler.fetch_page_text(url, cfg, prose_only=False)
            if page["text"].strip():
                pages.append(page)
            else:
                skipped.append(url)
        except crawler.CrawlError as exc:
            if log:
                log(f"  skipped: {exc}")
            skipped.append(url)
    if pages:
        return pages, skipped

    results = websearch.ddg_search(query, n or cfg["web_results"], cfg)
    for r in results[:4]:
        try:
            if log:
                log(f"fetching {r['url']}...")
            page = crawler.fetch_page_text(r["url"], cfg)
            if page["text"].strip():
                pages.append(page)
        except crawler.CrawlError as exc:
            if log:
                log(f"  skipped: {exc}")
            skipped.append(r["url"])
        if len(pages) >= 3:
            break
    if not pages:
        raise NoContext("Could not fetch any result pages.")
    return pages, skipped


def web_ask(query: str, cfg: dict, n: int | None = None, log: Log | None = None,
            stream_to: llm.StreamFn | None = None) -> dict:
    pages, skipped = fetch_web_pages(query, cfg, n=n, log=log)
    answer = llm.answer(query, pages, cfg, stream_to=stream_to,
                        limit=web_source_limit(pages))
    return {"query": query, "answer": answer,
            "sources": _sources(pages, text_limit=1200), "skipped": skipped}


def crawl_site(conn: sqlite3.Connection, url: str, cfg: dict, depth: int = 1,
               max_pages: int = 30, same_domain: bool = True, log: Log | None = None,
               on_page: Callable[[dict, int], None] | None = None,
               should_stop: Callable[[], bool] | None = None,
               path_prefix: str | None = None) -> dict:
    """Crawl a site into the index. `should_stop` is checked after every page."""
    pages = chunks = 0
    for page in crawler.crawl(url, cfg, depth=depth, max_pages=max_pages,
                              same_domain=same_domain, log=log or (lambda s: None),
                              path_prefix=path_prefix):
        n = indexer.add_document(conn, page["url"], page["title"], page["text"], "web", cfg)
        pages += 1
        chunks += n
        if on_page:
            on_page(page, n)
        if should_stop and should_stop():
            break
    return {"pages": pages, "chunks": chunks}
