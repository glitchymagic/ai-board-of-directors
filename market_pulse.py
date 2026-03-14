"""
Market Pulse — Multi-model market sentiment consensus.

Sends the same market snapshot to 3 different AI models in parallel,
extracts sentiment (BULLISH/BEARISH/NEUTRAL) from each, and computes
a consensus view with agreement percentage.

Usage:
    from market_pulse import run_pulse

    result = run_pulse("BTC: $70,000, F&G: 65 (Greed), Funding: 0.01%...")
    print(f"Consensus: {result['consensus']}, Agreement: {result['agreement_pct']}%")
"""

import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from ai_client import AIClient

DEFAULT_MODELS = [
    ("claude-sonnet-4", "Claude Sonnet 4"),
    ("gpt-5.4-medium", "GPT-5.4 Medium"),
    ("gemini-3.1-pro", "Gemini 3.1 Pro"),
]

PULSE_PROMPT = """You are a crypto/stock market analyst. Given this market snapshot, provide a concise 2-3 sentence take on the current market conditions and what traders should watch for in the next 6-12 hours.

MARKET SNAPSHOT:
{snapshot}

Be specific, direct, and actionable. No disclaimers. Focus on what matters most RIGHT NOW. End with a one-word sentiment: BULLISH, BEARISH, or NEUTRAL."""

HISTORY_FILE = Path("./data/market_pulse_history.json")
PULSE_FILE = Path("./data/market_pulse.json")


def synthesize_consensus(responses: list[dict | None], models: list[tuple[str, str]]) -> dict:
    """Extract sentiment from each response and compute consensus."""
    takes = []
    sentiments = []

    for i, resp in enumerate(responses):
        model_name = models[i][1] if i < len(models) else "Unknown"
        if not resp:
            takes.append({"model": model_name, "take": "unavailable", "sentiment": "NEUTRAL", "latency_ms": 0})
            continue

        content = resp["content"]
        # Extract sentiment from last non-empty line
        sentiment = "NEUTRAL"
        for line in reversed(content.upper().split("\n")):
            if line.strip():
                for word in ["BEARISH", "BULLISH", "NEUTRAL"]:
                    if word in line:
                        sentiment = word
                        break
                break

        sentiments.append(sentiment)
        takes.append({
            "model": model_name,
            "take": content,
            "sentiment": sentiment,
            "latency_ms": resp["latency_ms"],
        })

    # Consensus
    if sentiments:
        most_common = Counter(sentiments).most_common(1)[0]
        consensus = most_common[0]
        agreement = most_common[1] / len(sentiments) * 100
    else:
        consensus = "NEUTRAL"
        agreement = 0

    return {
        "consensus": consensus,
        "agreement_pct": round(agreement, 0),
        "takes": takes,
    }


def format_summary(result: dict) -> str:
    """Format the market pulse as a readable summary."""
    lines = [
        f"Market Pulse: {result['consensus']} ({result['agreement_pct']:.0f}% agreement)",
        "",
    ]

    for take in result["takes"]:
        if take["take"] == "unavailable":
            lines.append(f"{take['model']}: unavailable")
        else:
            text = take["take"]
            if len(text) > 500:
                text = text[:497] + "..."
            lines.append(f"{take['model']} ({take['latency_ms']}ms):")
            lines.append(text)
        lines.append("")

    return "\n".join(lines)


def run_pulse(snapshot: str, models: list[tuple[str, str]] | None = None) -> dict:
    """Run a market pulse analysis with multi-model consensus.

    Args:
        snapshot: Market data snapshot string that all models will analyze.
        models: List of (model_id, display_name) tuples. Defaults to
                Claude Sonnet 4, GPT-5.4 Medium, Gemini 3.1 Pro.

    Returns:
        Dict with keys: consensus, agreement_pct, takes, timestamp, snapshot.
    """
    if models is None:
        models = DEFAULT_MODELS

    now = datetime.now(UTC)
    print(f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] Market pulse starting...")
    print(f"Snapshot:\n{snapshot}\n")

    prompt = PULSE_PROMPT.format(snapshot=snapshot)
    messages = [{"role": "user", "content": prompt}]

    client = AIClient()

    # Check health first
    health = client.check_health()
    if not health["fast"] and not health["multi"]:
        print("Both AI proxies are down — skipping pulse.")
        return {"error": "proxy_down", "consensus": "NEUTRAL", "agreement_pct": 0, "takes": []}

    # Send to each model sequentially for reliability
    responses = []
    for model_id, model_name in models:
        print(f"  Querying {model_name}...")
        resp = client.chat(
            model=model_id, messages=messages,
            max_tokens=400, temperature=0.4, timeout=90,
        )
        responses.append(resp)

    successful = sum(1 for r in responses if r is not None)
    print(f"Got {successful}/{len(models)} responses")

    if successful == 0:
        print("No model responded — skipping pulse.")
        return {"error": "no_responses", "consensus": "NEUTRAL", "agreement_pct": 0, "takes": []}

    result = synthesize_consensus(responses, models)
    result["timestamp"] = now.isoformat()
    result["snapshot"] = snapshot

    # Save to file (keep last 30 pulses for history)
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        history = json.loads(HISTORY_FILE.read_text()) if HISTORY_FILE.exists() else []
    except (json.JSONDecodeError, OSError):
        history = []
    history.append({"timestamp": result["timestamp"], "consensus": result["consensus"], "agreement_pct": result["agreement_pct"]})
    history = history[-30:]
    tmp_hist = str(HISTORY_FILE) + ".tmp"
    with open(tmp_hist, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp_hist, str(HISTORY_FILE))

    # Save current pulse
    tmp = str(PULSE_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(result, f, indent=2, default=str)
    os.replace(tmp, str(PULSE_FILE))

    # Print summary
    summary = format_summary(result)
    print(summary)
    print(f"Saved to {PULSE_FILE}")

    return result


if __name__ == "__main__":
    import sys

    if not sys.stdin.isatty():
        snapshot_text = sys.stdin.read()
    else:
        snapshot_text = "No snapshot provided. Provide market data via stdin."

    run_pulse(snapshot_text)
