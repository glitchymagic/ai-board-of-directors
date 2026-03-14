"""
AI Client — Lightweight wrapper for OpenAI-compatible AI proxy endpoints.

Supports two proxy tiers:
- Fast proxy (e.g., port 3456): Claude Opus, Sonnet, Haiku. Low latency (3-7s).
- Multi-model proxy (e.g., port 8765): 44+ models — GPT, Gemini, Grok, etc. (10-17s).

Both use the OpenAI /v1/chat/completions format.

Usage:
    from ai_client import AIClient
    client = AIClient()
    result = client.chat("claude-haiku-4", [{"role": "user", "content": "Hello"}])
    if result:
        print(result["content"])
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# Proxy URLs — configurable via environment variables
FAST_PROXY = os.environ.get("AI_PROXY_FAST", "http://localhost:3456/v1")
MULTI_PROXY = os.environ.get("AI_PROXY_MULTI", "http://localhost:8765/v1")

# Models routed to the fast proxy
_FAST_MODELS = {"claude-haiku-4", "claude-sonnet-4", "claude-opus-4"}


def _route_model(model: str) -> str:
    """Return the correct proxy base URL for a given model."""
    if model in _FAST_MODELS:
        return FAST_PROXY
    return MULTI_PROXY


class AIClient:
    """Stateless AI proxy client. All methods return None on failure."""

    def chat(
        self,
        model: str,
        messages: list,
        max_tokens: int = 500,
        temperature: float = 0.3,
        timeout: int | None = None,
    ) -> dict | None:
        """Send a chat completion request.

        Returns {"content": str, "model": str, "latency_ms": int} or None.
        """
        base_url = _route_model(model)
        if timeout is None:
            timeout = 15 if model in _FAST_MODELS else 30

        for attempt in range(2):
            start = time.time()
            try:
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": "Bearer dummy"},
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    timeout=timeout,
                )
                if resp.status_code != 200:
                    print(f"[ai_client] {model}: HTTP {resp.status_code}")
                    if attempt == 0:
                        time.sleep(2)
                        continue
                    return None

                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                latency_ms = int((time.time() - start) * 1000)

                return {
                    "content": content,
                    "model": data.get("model", model),
                    "latency_ms": latency_ms,
                }
            except requests.Timeout:
                print(f"[ai_client] {model}: timeout after {timeout}s (attempt {attempt + 1})")
                if attempt == 0:
                    continue
                return None
            except (requests.RequestException, KeyError, IndexError, ValueError, TypeError) as e:
                print(f"[ai_client] {model}: {type(e).__name__}: {e}")
                return None
        return None

    def chat_parallel(self, reqs: list[dict]) -> list[dict | None]:
        """Send multiple chat requests in parallel.

        Each item in reqs should be a dict with keys: model, messages,
        and optionally max_tokens, temperature, timeout.

        Returns list of results in same order as input.
        """
        if not reqs:
            return []

        results = [None] * len(reqs)

        def _do(idx, req):
            return idx, self.chat(
                model=req["model"],
                messages=req["messages"],
                max_tokens=req.get("max_tokens", 500),
                temperature=req.get("temperature", 0.3),
                timeout=req.get("timeout"),
            )

        with ThreadPoolExecutor(max_workers=min(len(reqs), 5)) as pool:
            futures = [pool.submit(_do, i, r) for i, r in enumerate(reqs)]
            for future in as_completed(futures):
                try:
                    idx, result = future.result()
                    results[idx] = result
                except Exception:
                    pass

        return results

    def check_health(self) -> dict:
        """Check which proxies are reachable.

        Returns {"fast": bool, "multi": bool}.
        """
        status = {"fast": False, "multi": False}

        for key, url in [("fast", FAST_PROXY), ("multi", MULTI_PROXY)]:
            try:
                resp = requests.get(f"{url}/models", timeout=5)
                status[key] = resp.status_code == 200
            except requests.RequestException:
                pass

        return status
