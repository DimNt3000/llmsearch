"""Polite stdlib-only web crawler: fetch, extract text/links, BFS crawl.

Respects robots.txt, stays on the start domain by default, and throttles
requests. Uses only urllib + html.parser so it has zero dependencies.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.request
from html import unescape
from html.parser import HTMLParser
from typing import Callable, Iterator
from urllib.parse import urldefrag, urljoin, urlparse

MAX_BYTES = 3 * 1024 * 1024  # per-page download cap

# Text inside these is never indexed. Menus and footers are boilerplate that
# repeats on every page: a documentation sidebar listing the whole API made up
# 80% of each PyTorch page. Links inside them are still followed.
_SKIP_CONTENT = frozenset(
    ("script", "style", "noscript", "template", "svg", "iframe", "nav", "footer"))
_BLOCK_TAGS = frozenset(
    ("p", "div", "br", "li", "ul", "ol", "table", "tr", "section", "article",
     "header", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre")
)


class CrawlError(Exception):
    pass


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_CONTENT:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_CONTENT and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.text_parts.append("\n\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        if self._in_title:
            self.title_parts.append(data)
        elif data.strip():
            self.text_parts.append(data)


def fetch(url: str, ua: str, timeout: float) -> tuple[str, str, bytes]:
    """GET a URL. Returns (final_url, content_type, decompressed body)."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": ua,
            "Accept": "text/html,text/plain,application/xhtml+xml,*/*;q=0.5",
            "Accept-Language": "en,el;q=0.8,*;q=0.5",
            "Accept-Encoding": "gzip, identity",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            final_url = resp.geturl()
            ctype = resp.headers.get("Content-Type", "")
            body = resp.read(MAX_BYTES)
            encoding = (resp.headers.get("Content-Encoding") or "").lower()
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise CrawlError(f"{url}: {exc}") from exc
    return final_url, ctype, _decompress(body, encoding, url)


def _decompress(body: bytes, encoding: str, url: str) -> bytes:
    if "gzip" in encoding or body[:2] == b"\x1f\x8b":
        import gzip
        try:
            return gzip.decompress(body)
        except OSError as exc:
            raise CrawlError(f"{url}: bad gzip body: {exc}") from exc
    if "deflate" in encoding:
        import zlib
        try:
            return zlib.decompress(body)
        except zlib.error:
            try:
                return zlib.decompress(body, -zlib.MAX_WBITS)
            except zlib.error as exc:
                raise CrawlError(f"{url}: bad deflate body: {exc}") from exc
    if "br" in encoding or "zstd" in encoding:
        raise CrawlError(f"{url}: unsupported content-encoding {encoding!r}")
    return body


def decode(body: bytes, content_type: str) -> str:
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=", 1)[1].split(";")[0].strip().strip('"')
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def extract(html: str, base_url: str) -> tuple[str, str, list[str]]:
    """Parse HTML into (title, text, absolute_links)."""
    parser = _PageParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass  # salvage whatever was parsed before the error
    title = unescape(" ".join("".join(parser.title_parts).split()))
    text = "\n".join(
        line.strip() for line in "".join(parser.text_parts).splitlines()
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    links: list[str] = []
    seen: set[str] = set()
    for href in parser.links:
        absolute = urldefrag(urljoin(base_url, href)).url
        if urlparse(absolute).scheme in ("http", "https") and absolute not in seen:
            seen.add(absolute)
            links.append(absolute)
    return title, text, links


RobotsRules = list[tuple[bool, "re.Pattern[str]", int]]  # (allow, pattern, length)


def _robots_pattern(path: str) -> "re.Pattern[str]":
    """A robots.txt path as a regex: `*` matches any run, a final `$` anchors."""
    anchored = path.endswith("$")
    body = re.escape(path[:-1] if anchored else path).replace(r"\*", ".*")
    return re.compile(body + ("$" if anchored else ""))


def parse_robots(text: str, token: str) -> RobotsRules:
    """The rules a robots.txt sets for the crawler whose product token is `token`.

    As in RFC 9309, the rules come from every group that names the token, or
    from the `*` groups when none does, and consecutive user-agent lines share
    one group.
    """
    groups: list[tuple[set[str], list[tuple[bool, str]]]] = []
    agents: set[str] = set()
    rules: list[tuple[bool, str]] = []
    in_rules = False
    for raw in text.splitlines():
        key, sep, value = raw.split("#", 1)[0].partition(":")
        if not sep:
            continue
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            if in_rules:  # a user-agent line after rules starts the next group
                groups.append((agents, rules))
                agents, rules, in_rules = set(), [], False
            agents.add(value.split("/")[0].strip().lower())
        elif key in ("allow", "disallow") and agents:
            in_rules = True
            if value:  # an empty Disallow allows everything, so it adds no rule
                rules.append((key == "allow", value))
    if agents:
        groups.append((agents, rules))
    chosen = [g for g in groups if token in g[0]] or [g for g in groups if "*" in g[0]]
    return [(allow, _robots_pattern(path), len(path))
            for _, group_rules in chosen for allow, path in group_rules]


class _Robots:
    """robots.txt rules per origin, fetched with the crawler's own user agent.

    urllib.robotparser fetches robots.txt with Python's default user agent,
    which bot protection such as Cloudflare answers with 403, and it reads a
    403 as "disallow everything"; it also takes `*` and `$` literally. This
    follows RFC 9309 instead: the longest matching rule wins and Allow wins a
    tie, a missing or forbidden robots.txt (4xx) allows everything, and a
    server error (5xx) allows nothing.
    """

    def __init__(self, ua: str, timeout: float) -> None:
        self.ua = ua
        self.timeout = timeout
        self.token = (ua.split("/")[0].split() or ["*"])[0].lower()
        self._cache: dict[str, bool | RobotsRules] = {}

    def _load(self, origin: str) -> bool | RobotsRules:
        try:
            _, ctype, body = fetch(origin + "/robots.txt", self.ua, self.timeout)
        except CrawlError as exc:
            status = getattr(exc.__cause__, "code", None)
            return not (isinstance(status, int) and status >= 500)
        return parse_robots(decode(body, ctype), self.token)

    def allowed(self, url: str) -> bool:
        parts = urlparse(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._cache:
            self._cache[origin] = self._load(origin)
        rules = self._cache[origin]
        if isinstance(rules, bool):
            return rules
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        best, verdict = -1, True
        for allow, pattern, length in rules:
            if pattern.match(target) and (length > best or (length == best and allow)):
                best, verdict = length, allow
        return verdict


def crawl(start_url: str, cfg: dict, depth: int = 1, max_pages: int = 30,
          same_domain: bool = True,
          log: Callable[[str], None] = lambda s: None,
          path_prefix: str | None = None) -> Iterator[dict]:
    """BFS crawl yielding {"url", "title", "text"} per fetched page.

    `path_prefix` keeps the crawl inside one section of a site, such as
    "/docs/2.14/", by following only links whose path starts with it.
    """
    ua, timeout, delay = cfg["user_agent"], cfg["http_timeout"], cfg["crawl_delay"]
    robots = _Robots(ua, timeout)
    start_host = urlparse(start_url).netloc.lower()
    queue: list[tuple[str, int]] = [(start_url, 0)]
    seen: set[str] = {start_url}
    fetched = 0

    while queue and fetched < max_pages:
        url, level = queue.pop(0)
        if not robots.allowed(url):
            log(f"  skip (robots.txt): {url}")
            continue
        try:
            final_url, ctype, body = fetch(url, ua, timeout)
        except CrawlError as exc:
            log(f"  error: {exc}")
            continue

        main_type = ctype.split(";")[0].strip().lower()
        if main_type not in ("text/html", "application/xhtml+xml", "text/plain", ""):
            log(f"  skip ({main_type or 'unknown type'}): {url}")
            continue

        text_raw = decode(body, ctype)
        if main_type == "text/plain":
            title, text, links = url.rsplit("/", 1)[-1] or url, text_raw, []
        else:
            title, text, links = extract(text_raw, final_url)

        fetched += 1
        if text:
            yield {"url": final_url, "title": title or final_url, "text": text}
        else:
            log(f"  no text: {url}")

        if level < depth:
            for link in links:
                if link in seen:
                    continue
                if same_domain and urlparse(link).netloc.lower() != start_host:
                    continue
                if path_prefix and not urlparse(link).path.startswith(path_prefix):
                    continue
                seen.add(link)
                queue.append((link, level + 1))

        if queue and fetched < max_pages:
            time.sleep(delay)


def _prose_only(text: str) -> str:
    """Drop navigation/menu noise: keep lines that read like sentences."""
    good = [line for line in text.splitlines() if len(line.split()) >= 8]
    return "\n".join(good)


def fetch_page_text(url: str, cfg: dict, max_chars: int = 9000) -> dict:
    """Fetch a single page and return {"url", "title", "text"} (for `web --ask`)."""
    final_url, ctype, body = fetch(url, cfg["user_agent"], cfg["http_timeout"])
    text_raw = decode(body, ctype)
    if ctype.split(";")[0].strip().lower() == "text/plain":
        title, text = url, text_raw
    else:
        title, text, _ = extract(text_raw, final_url)
        prose = _prose_only(text)
        if len(prose) >= 400:  # fall back to raw text on sparse pages
            text = prose
    return {"url": final_url, "title": title or final_url, "text": text[:max_chars]}
