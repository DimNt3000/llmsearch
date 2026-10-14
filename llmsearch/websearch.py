"""Live web search via DuckDuckGo.

Prefers the `ddgs` package when installed (more reliable); falls back to
parsing DuckDuckGo's HTML endpoint with the stdlib. Either way, results
are normalized to [{"title", "url", "snippet"}].
"""

from __future__ import annotations

import re
import urllib.parse
from html import unescape
from html.parser import HTMLParser

from . import crawler


class WebSearchError(Exception):
    pass


# A bare name like "sport24.gr" counts as a site only with one of these
# endings, so "run.py", "Node.js" or "e.g." in a question are not fetched.
# A scheme or "www." makes any ending count.
_SITE_TLDS = frozenset("""
    com org net edu gov int mil io dev ai app co info biz me tv fm news blog
    site online tech cloud page wiki xyz gr cy uk us eu de fr it es nl be ch at
    se no dk fi is ie pt pl cz sk hu ro bg hr si rs ru ua tr il jp cn kr in sg
    hk tw au nz ca mx br ar cl za
""".split())
_URL_RE = re.compile(
    r"(?<![\w@./-])"
    r"((?:https?://)?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+([a-z]{2,24})"
    r"(?::\d{2,5})?(?:/[^\s\"'<>«»]*)?)",
    re.IGNORECASE,
)
_TRAILING = ".,;:!?)]}…·'\"»"  # sentence punctuation, Greek ";" and "·" included


def direct_urls(query: str, limit: int = 3) -> list[str]:
    """Sites named in a question ("what does sport24.gr say now?"), as URLs."""
    urls: list[str] = []
    for m in _URL_RE.finditer(query):
        text = m.group(1).rstrip(_TRAILING)
        explicit = text.lower().startswith(("http://", "https://", "www."))
        if not explicit and m.group(2).lower() not in _SITE_TLDS:
            continue
        url = text if text.lower().startswith(("http://", "https://")) else "https://" + text
        if url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def ddg_search(query: str, n: int, cfg: dict) -> list[dict]:
    try:
        return _search_with_ddgs(query, n)
    except ImportError:
        return _search_html_fallback(query, n, cfg)


def _search_with_ddgs(query: str, n: int) -> list[dict]:
    try:
        from ddgs import DDGS  # current package name
    except ImportError:
        from duckduckgo_search import DDGS  # legacy package name

    try:
        with DDGS() as client:
            raw = list(client.text(query, max_results=n))
    except Exception as exc:  # ddgs raises library-specific errors
        raise WebSearchError(f"web search failed: {exc}") from exc

    results = []
    for item in raw:
        url = item.get("href") or item.get("url") or ""
        if not url:
            continue
        results.append({
            "title": unescape(item.get("title") or url),
            "url": url,
            "snippet": unescape(item.get("body") or item.get("snippet") or ""),
        })
    return results[:n]


class _DDGHtmlParser(HTMLParser):
    """Extracts result links/snippets from html.duckduckgo.com/html."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._mode: str | None = None  # "title" | "snippet"
        self._buffer: list[str] = []
        self._href = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "")
        if tag == "a" and "result__a" in cls:
            self._mode, self._buffer, self._href = "title", [], a.get("href", "")
        elif "result__snippet" in cls:
            self._mode, self._buffer = "snippet", []

    def handle_endtag(self, tag):
        if self._mode == "title" and tag == "a":
            self.results.append({
                "title": " ".join("".join(self._buffer).split()),
                "url": _clean_ddg_url(self._href),
                "snippet": "",
            })
            self._mode = None
        elif self._mode == "snippet" and tag in ("a", "td", "div", "span"):
            if self.results:
                self.results[-1]["snippet"] = " ".join("".join(self._buffer).split())
            self._mode = None

    def handle_data(self, data):
        if self._mode:
            self._buffer.append(data)


def _clean_ddg_url(href: str) -> str:
    """DDG wraps results as //duckduckgo.com/l/?uddg=<encoded>&rut=..."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        params = urllib.parse.parse_qs(parsed.query)
        if "uddg" in params:
            return params["uddg"][0]
    return href


def _search_html_fallback(query: str, n: int, cfg: dict) -> list[dict]:
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote_plus(query)
    try:
        _, ctype, body = crawler.fetch(url, cfg["user_agent"], cfg["http_timeout"])
    except crawler.CrawlError as exc:
        raise WebSearchError(f"web search failed: {exc}") from exc

    html = crawler.decode(body, ctype)
    parser = _DDGHtmlParser()
    parser.feed(html)
    results = [r for r in parser.results if r["url"]][:n]
    if not results:
        hint = ("DuckDuckGo returned no parseable results (possibly a bot "
                "challenge). Install the `ddgs` package for reliable web "
                "search: pip install ddgs")
        raise WebSearchError(hint)
    return results
