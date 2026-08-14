"""LLM layer: Claude via the Anthropic SDK, with a `claude` CLI fallback.

Backends
  api        Anthropic Python SDK (needs ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN,
             or an `ant auth login` profile). Default model: claude-opus-5.
             Server-side refusal fallbacks are requested by default.
  claude-cli Shells out to the Claude Code CLI (`claude -p`), reusing its login.
  auto       api when credentials look available, otherwise claude-cli.
  off        Disables LLM features (plain search keeps working).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


class LLMError(Exception):
    pass


class LLMUnavailable(LLMError):
    """No usable backend; the message explains how to enable one."""


StreamFn = Callable[[str], None]

_SETUP_HINT = (
    "No LLM backend available. Either:\n"
    "  - set ANTHROPIC_API_KEY (in the environment or in <project>/.env), or\n"
    "  - install Claude Code so the `claude` CLI is on PATH, or\n"
    "  - pip install anthropic + `ant auth login`.\n"
    "Plain `search`, `crawl`, `add`, and `web` work without an LLM."
)


def _api_importable() -> bool:
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def _api_creds_likely() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    # An `ant auth login` profile also authenticates the SDK.
    if sys.platform == "win32":
        profile_dir = Path(os.environ.get("APPDATA", "")) / "Anthropic"
    else:
        profile_dir = Path.home() / ".config" / "anthropic"
    return (profile_dir / "credentials").is_dir()


def resolve_backend(cfg: dict) -> str:
    backend = cfg.get("backend", "auto")
    if backend != "auto":
        return backend
    if _api_importable() and _api_creds_likely():
        return "api"
    if shutil.which("claude"):
        return "claude-cli"
    if _api_importable():
        return "api"  # last try: the SDK may find credentials we can't see
    return "off"


def complete(system: str, prompt: str, cfg: dict, max_tokens: int | None = None,
             stream_to: StreamFn | None = None) -> str:
    """Run one completion on the resolved backend. Returns the full text."""
    backend = resolve_backend(cfg)
    if backend == "off":
        raise LLMUnavailable(_SETUP_HINT)
    if backend == "api":
        try:
            return _api_complete(system, prompt, cfg, max_tokens, stream_to)
        except LLMUnavailable:
            if cfg.get("backend", "auto") == "auto" and shutil.which("claude"):
                return _cli_complete(system, prompt, cfg, stream_to)
            raise
    if backend == "claude-cli":
        return _cli_complete(system, prompt, cfg, stream_to)
    raise LLMError(f"unknown backend: {backend!r}")


# --------------------------------------------------------------------------
# Anthropic API backend
# --------------------------------------------------------------------------

def _api_complete(system: str, prompt: str, cfg: dict,
                  max_tokens: int | None, stream_to: StreamFn | None) -> str:
    import anthropic

    client = anthropic.Anthropic()
    kwargs = dict(
        model=cfg["model"],
        max_tokens=max_tokens or cfg["max_answer_tokens"],
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    # Server-side refusal fallbacks (recommended default for claude-opus-5).
    beta_kwargs = dict(betas=["server-side-fallback-2026-07-01"], fallbacks="default")

    try:
        try:
            return _api_call(client, kwargs | beta_kwargs, stream_to, beta=True)
        except (TypeError, anthropic.BadRequestError):
            # Older SDK/API without the fallbacks parameter: plain call.
            return _api_call(client, kwargs, stream_to, beta=False)
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable(f"Anthropic API auth failed ({exc.message}).\n{_SETUP_HINT}") from exc
    except anthropic.RateLimitError as exc:
        raise LLMError("Anthropic API rate limit hit - wait a bit and retry.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError(f"cannot reach the Anthropic API: {exc}") from exc


def _api_call(client, kwargs: dict, stream_to: StreamFn | None, beta: bool) -> str:
    endpoint = client.beta.messages if beta else client.messages
    if stream_to is not None:
        with endpoint.stream(**kwargs) as stream:
            for text in stream.text_stream:
                stream_to(text)
            message = stream.get_final_message()
    else:
        message = endpoint.create(**kwargs)

    if message.stop_reason == "refusal":
        detail = ""
        details = getattr(message, "stop_details", None)
        if details is not None and getattr(details, "explanation", None):
            detail = f" ({details.explanation})"
        raise LLMError(f"the model declined this request{detail}")
    return "".join(b.text for b in message.content if b.type == "text").strip()


# --------------------------------------------------------------------------
# Claude Code CLI backend
# --------------------------------------------------------------------------

def _cli_complete(system: str, prompt: str, cfg: dict,
                  stream_to: StreamFn | None) -> str:
    exe = shutil.which("claude")
    if not exe:
        raise LLMUnavailable(_SETUP_HINT)
    full_prompt = f"{system}\n\n---\n\n{prompt}" if system else prompt
    try:
        proc = subprocess.run(
            [exe, "-p", "--model", cfg.get("cli_model", "opus"), "--output-format", "text"],
            input=full_prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=cfg.get("cli_timeout", 240),
        )
    except subprocess.TimeoutExpired as exc:
        raise LLMError(f"claude CLI timed out after {cfg.get('cli_timeout', 240)}s") from exc
    except OSError as exc:
        raise LLMUnavailable(f"could not run the claude CLI: {exc}\n{_SETUP_HINT}") from exc

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip() or "unknown error"
        raise LLMError(f"claude CLI failed (exit {proc.returncode}): {stderr[:400]}")
    result = (proc.stdout or "").strip()
    if not result:
        raise LLMError("claude CLI returned no output")
    if stream_to is not None:
        stream_to(result)
    return result


# --------------------------------------------------------------------------
# Search-engine features built on complete()
# --------------------------------------------------------------------------

def _format_sources(sources: list[dict], text_key: str, limit: int = 1800) -> str:
    blocks = []
    for i, s in enumerate(sources, 1):
        body = " ".join((s.get(text_key) or "").split())[:limit]
        blocks.append(f"[{i}] {s.get('title', 'untitled')}\nURL: {s.get('url', '?')}\n{body}")
    return "\n\n".join(blocks)


def answer(question: str, sources: list[dict], cfg: dict,
           stream_to: StreamFn | None = None) -> str:
    system = (
        "You are the answer engine of a small local search tool. Answer the "
        "user's question using ONLY the numbered sources provided. Cite sources "
        "inline as [n] after each claim. If the sources do not contain the "
        f"answer, say so plainly. Style: {cfg['answer_style']}. No preamble."
    )
    prompt = f"Question: {question}\n\nSources:\n\n{_format_sources(sources, 'text')}"
    return complete(system, prompt, cfg, stream_to=stream_to)


def summarize(query: str, results: list[dict], cfg: dict,
              stream_to: StreamFn | None = None) -> str:
    system = (
        "You are the summarization layer of a search engine. Given a query and "
        "search results, write a short digest: what the results collectively "
        "say, key points as brief bullets, citing results as [n]. "
        f"Style: {cfg['answer_style']}. No preamble."
    )
    key = "text" if results and "text" in results[0] else "snippet"
    prompt = f"Query: {query}\n\nResults:\n\n{_format_sources(results, key, limit=900)}"
    return complete(system, prompt, cfg, stream_to=stream_to)


def expand_query(query: str, cfg: dict, n: int = 3) -> list[str]:
    system = (
        "You expand search queries for a keyword-based (BM25) search engine. "
        "Return ONLY a JSON array of strings, no other text."
    )
    prompt = (
        f"Give {n} alternative queries for: {query!r}. Use synonyms, "
        "rephrasings, and related technical terms. Keep each under 8 words."
    )
    try:
        raw = complete(system, prompt, cfg, max_tokens=300)
        variants = _extract_json_array(raw)
        return [v for v in variants if isinstance(v, str) and v.strip()][:n]
    except LLMError:
        raise
    except Exception:
        return []


def rerank(query: str, results: list[dict], cfg: dict, keep: int) -> list[dict]:
    """Ask the model to reorder results by relevance; fail open to BM25 order."""
    if len(results) <= 1:
        return results
    system = (
        "You rerank search results by relevance to a query. Return ONLY a JSON "
        "array of 0-based indices, most relevant first, no other text."
    )
    listing = "\n".join(
        f"{i}. {r.get('title', '?')} - "
        + " ".join((r.get("text") or r.get("snippet") or "").split())[:200]
        for i, r in enumerate(results)
    )
    prompt = (f"Query: {query!r}\n\nResults:\n{listing}\n\n"
              f"Return the indices of the {min(keep, len(results))} best results in order.")
    try:
        raw = complete(system, prompt, cfg, max_tokens=200)
        order = _extract_json_array(raw)
        picked, seen = [], set()
        for idx in order:
            if isinstance(idx, int) and 0 <= idx < len(results) and idx not in seen:
                seen.add(idx)
                picked.append(results[idx])
        return picked[:keep] if picked else results[:keep]
    except LLMError:
        raise
    except Exception:
        return results[:keep]


def _extract_json_array(raw: str) -> list:
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON array in model output: {raw[:200]!r}")
    return json.loads(match.group(0))
