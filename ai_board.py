"""
AI Board of Directors — Multi-model consensus with role-specialized expert agents.

6 AI models with distinct professional roles evaluate the same briefing in parallel.
Each provides a structured assessment (grade, confidence, top concern, recommendation).
Results are synthesized into a consensus and persisted to board_history.json.

Usage:
    from ai_board import run_board

    result = run_board("Your briefing here...")
    print(f"Grade: {result['consensus_grade']}, Agreement: {result['agreement_pct']}%")

    # Agenda vote (yes/no decision)
    result = run_board(briefing, agenda="Should we proceed with X?")
"""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

from ai_client import AIClient

HISTORY_FILE = Path("./data/board_history.json")

# Board members — each has a role that shapes their system prompt
BOARD_MEMBERS = [
    {
        "name": "Risk Officer",
        "model": "opus-4.5",
        "role": (
            "You are the RISK OFFICER on an AI Board of Directors. "
            "Your focus: drawdown risk, position sizing, correlation exposure, tail risk, "
            "worst-case scenarios, and whether the system can survive black swan events. "
            "Be conservative — your job is to find what could blow up."
        ),
    },
    {
        "name": "Quant Analyst",
        "model": "sonnet-4.6-thinking",
        "role": (
            "You are the QUANT ANALYST on an AI Board of Directors. "
            "Your focus: statistical edge, setup performance, sample size significance, "
            "regime sensitivity, overfitting risk, and whether the numbers actually prove an edge. "
            "Be rigorous — demand statistical evidence."
        ),
    },
    {
        "name": "Infra Engineer",
        "model": "gpt-5.4-high",
        "role": (
            "You are the INFRASTRUCTURE ENGINEER on an AI Board of Directors. "
            "Your focus: system reliability, deployment architecture, failure modes, monitoring, "
            "data integrity, and whether the infrastructure can handle real-money operations. "
            "Think about what breaks at 3am with no one watching."
        ),
    },
    {
        "name": "Devil's Advocate",
        "model": "grok",
        "role": (
            "You are the DEVIL'S ADVOCATE on an AI Board of Directors. "
            "Your job is to challenge every assumption, find flaws the team is blind to, "
            "argue against going live, and stress-test the team's confidence. "
            "Be skeptical and provocative — if you can't find problems, you're not looking hard enough."
        ),
    },
    {
        "name": "Strategist",
        "model": "gemini-3.1-pro",
        "role": (
            "You are the STRATEGIST on an AI Board of Directors. "
            "Your focus: roadmap priorities, capital allocation strategy, market positioning, "
            "the paper-to-real transition plan, and what to build vs what to cut. "
            "Think about ROI of effort and what moves the needle most."
        ),
    },
    {
        "name": "Fund Manager",
        "model": "gpt-5.4-medium",
        "role": (
            "You are the FUND MANAGER on an AI Board of Directors. "
            "Your focus: overall readiness for real capital, go/no-go decision, "
            "how much capital to deploy and when, performance benchmarks, and investor perspective. "
            "You make the final call — would you put your own money in this system?"
        ),
    },
]

STRUCTURED_FORMAT = """\
Respond in EXACTLY this format:

GRADE: [A+ to F]
CONFIDENCE: [1-10]
TOP_CONCERN: [One sentence — the single biggest issue from your role's perspective]
RECOMMENDATION: [2-3 sentences — your most actionable advice]

Then optionally add 1-2 paragraphs of supporting analysis. Keep total response under 350 words."""

AGENDA_FORMAT = """\
Respond in EXACTLY this format:

VOTE: [YES or NO]
CONFIDENCE: [1-10]
REASONING: [2-3 sentences explaining your vote from your role's perspective]

Keep total response under 150 words."""


def _parse_structured(content: str) -> dict:
    """Extract GRADE, CONFIDENCE, TOP_CONCERN, RECOMMENDATION from response."""
    parsed = {}

    grade_match = re.search(r"GRADE:\s*\**([A-F][+-]?)\**", content, re.IGNORECASE)
    if grade_match:
        parsed["grade"] = grade_match.group(1).upper()

    conf_match = re.search(r"CONFIDENCE:\s*(\d+)", content, re.IGNORECASE)
    if conf_match:
        parsed["confidence"] = min(int(conf_match.group(1)), 10)

    concern_match = re.search(r"TOP_CONCERN:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
    if concern_match:
        parsed["top_concern"] = concern_match.group(1).strip()

    rec_match = re.search(r"RECOMMENDATION:\s*(.+?)(?:\n\n|$)", content, re.IGNORECASE | re.DOTALL)
    if rec_match:
        parsed["recommendation"] = rec_match.group(1).strip()

    # Agenda vote parsing
    vote_match = re.search(r"VOTE:\s*(YES|NO)", content, re.IGNORECASE)
    if vote_match:
        parsed["vote"] = vote_match.group(1).upper()

    reason_match = re.search(r"REASONING:\s*(.+?)(?:\n\n|$)", content, re.IGNORECASE | re.DOTALL)
    if reason_match:
        parsed["reasoning"] = reason_match.group(1).strip()

    return parsed


def _append_history(meeting: dict):
    """Append meeting summary to board_history.json."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        history = json.loads(HISTORY_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    entry = {
        "date": meeting["timestamp"],
        "grades": meeting.get("grades", []),
        "confidences": meeting.get("confidences", []),
        "avg_confidence": meeting.get("avg_confidence"),
        "concerns": meeting.get("concerns", []),
        "member_count": meeting["responses_received"],
        "agenda": meeting.get("agenda"),
        "votes": meeting.get("votes"),
    }
    history.append(entry)

    tmp = str(HISTORY_FILE) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(history, f, indent=2)
    os.replace(tmp, str(HISTORY_FILE))


def run_board(briefing: str, send_telegram: bool = False, agenda: str | None = None) -> dict:
    """Run the board meeting. Returns structured results.

    Args:
        briefing: The briefing text all board members will evaluate.
        send_telegram: If True, print a Telegram-style summary (no actual sending).
        agenda: If provided, board members vote YES/NO on this specific question.

    Returns:
        Dict with keys: timestamp, board_size, responses_received, grades,
        confidences, avg_confidence, concerns, responses, and optionally
        agenda, votes, consensus_grade, agreement_pct.
    """
    client = AIClient()

    health = client.check_health()
    if not health.get("multi"):
        print("ERROR: Multi-model proxy is not reachable")
        return {"error": "proxy_down"}

    is_vote = agenda is not None

    if is_vote:
        briefing += f"\n## Agenda Item for Vote\n{agenda}\n"

    print(f"{'Agenda vote' if is_vote else 'Board meeting'} — {len(BOARD_MEMBERS)} members")
    print(f"Briefing: {len(briefing)} chars. Querying in parallel...\n")

    # Build per-member requests with role-specific system prompts
    output_format = AGENDA_FORMAT if is_vote else STRUCTURED_FORMAT
    reqs = []
    for member in BOARD_MEMBERS:
        messages = [
            {
                "role": "system",
                "content": f"{member['role']}\n\n{output_format}",
            },
            {"role": "user", "content": briefing},
        ]
        reqs.append({
            "model": member["model"],
            "messages": messages,
            "max_tokens": 600,
            "temperature": 0.4,
            "timeout": 90,
        })

    results = client.chat_parallel(reqs)

    # Process results
    responses = []
    grades = []
    confidences = []
    concerns = []
    votes = {"YES": [], "NO": []}

    for member, result in zip(BOARD_MEMBERS, results, strict=True):
        if result:
            parsed = _parse_structured(result["content"])
            resp = {
                "name": member["name"],
                "role": member["name"],
                "model": member["model"],
                "content": result["content"],
                "latency_ms": result["latency_ms"],
                **parsed,
            }
            responses.append(resp)

            if "grade" in parsed:
                grades.append(parsed["grade"])
            if "confidence" in parsed:
                confidences.append(parsed["confidence"])
            if "top_concern" in parsed:
                concerns.append(f"{member['name']}: {parsed['top_concern']}")
            if "vote" in parsed:
                votes[parsed["vote"]].append(member["name"])

            conf_str = f", conf {parsed.get('confidence', '?')}/10" if "confidence" in parsed else ""
            grade_str = f" [{parsed.get('grade', '?')}]" if "grade" in parsed else ""
            vote_str = f" [{parsed.get('vote', '?')}]" if "vote" in parsed else ""
            print(f"  {member['name']:20s}{grade_str}{vote_str}{conf_str} ({result['latency_ms']}ms)")
        else:
            print(f"  {member['name']:20s} FAILED")

    avg_conf = round(sum(confidences) / len(confidences), 1) if confidences else None

    # Compute consensus grade
    consensus_grade = None
    agreement_pct = 0
    if grades:
        from collections import Counter
        grade_counts = Counter(grades)
        most_common_grade, most_common_count = grade_counts.most_common(1)[0]
        consensus_grade = most_common_grade
        agreement_pct = round(most_common_count / len(grades) * 100, 0)

    print(f"\n{len(responses)}/{len(BOARD_MEMBERS)} responded")
    if grades:
        print(f"Grades: {', '.join(grades)}")
    if confidences:
        print(f"Avg confidence: {avg_conf}/10")
    if is_vote:
        print(f"Vote: {len(votes['YES'])} YES / {len(votes['NO'])} NO")

    # Build output
    output = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "board_size": len(BOARD_MEMBERS),
        "responses_received": len(responses),
        "grades": grades,
        "consensus_grade": consensus_grade,
        "agreement_pct": agreement_pct,
        "confidences": confidences,
        "avg_confidence": avg_conf,
        "concerns": concerns,
        "responses": responses,
    }
    if is_vote:
        output["agenda"] = agenda
        output["votes"] = votes

    # Print full responses
    print("\n" + "=" * 70)
    for r in responses:
        print(f"\n{'---' * 23}")
        print(f"  {r['name']} ({r['model']}, {r['latency_ms']}ms)")
        print(f"{'---' * 23}")
        print(r["content"])

    # Print summary if send_telegram flag is set
    if send_telegram and responses:
        lines = []
        if is_vote:
            lines.append(f"[Board Vote] {agenda}")
            lines.append(f"Result: {len(votes['YES'])} YES / {len(votes['NO'])} NO")
            if votes["YES"]:
                lines.append(f"YES: {', '.join(votes['YES'])}")
            if votes["NO"]:
                lines.append(f"NO: {', '.join(votes['NO'])}")
        else:
            lines.append(f"[Board Meeting] {len(responses)}/{len(BOARD_MEMBERS)} responded")
            if grades:
                lines.append(f"Grades: {', '.join(grades)}")
            if avg_conf:
                lines.append(f"Avg confidence: {avg_conf}/10")

        if concerns:
            lines.append("\nTop concerns:")
            for c in concerns:
                lines.append(f"  {c}")

        print("\n--- Summary ---")
        print("\n".join(lines))

    # Save current meeting
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_file = HISTORY_FILE.parent / "board_meeting.json"
    tmp = str(output_file) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(output, f, indent=2)
    os.replace(tmp, str(output_file))

    # Append to history
    _append_history(output)

    print(f"\nSaved to {output_file}")
    print(f"History appended to {HISTORY_FILE}")

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Board of Directors")
    parser.add_argument("--telegram", action="store_true", help="Print Telegram-style summary")
    parser.add_argument("--agenda", type=str, help="Put a specific question to a YES/NO vote")
    parser.add_argument("--briefing", type=str, help="Path to briefing text file (or pass via stdin)")
    args = parser.parse_args()

    if args.briefing:
        briefing_text = Path(args.briefing).read_text()
    else:
        import sys
        if not sys.stdin.isatty():
            briefing_text = sys.stdin.read()
        else:
            print("Provide a briefing via --briefing <file> or pipe to stdin.")
            sys.exit(1)

    run_board(briefing_text, send_telegram=args.telegram, agenda=args.agenda)
