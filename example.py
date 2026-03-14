"""Example: Running an AI Board of Directors meeting."""
from ai_board import run_board

# Define a briefing (this would come from your real data)
briefing = """
## System Status
- Portfolio: $6,467 (+29.3% from $5K start)
- Win Rate: 58.1% (86 closed trades)
- Open Positions: 3 (NFLX, PLTR, AMD)
- Drawdown: -7.4% (WARNING status)
- Soak Test: 6/12 gates GREEN

## Recent Performance
- Last 10 trades: 6W / 4L
- Best setup: BOS Trend (58.8% WR)
- Worst setup: Funding Squeeze (40% WR)

## Questions for the Board
1. Should we pause trading until drawdown recovers?
2. Should we remove Funding Squeeze setup?
"""

# Run a board meeting (requires AI proxy access)
result = run_board(briefing)
print(f"Consensus Grade: {result.get('consensus_grade', 'N/A')}")
print(f"Agreement: {result.get('agreement_pct', 0):.0f}%")

for member in result.get("responses", []):
    print(f"\n{member['role']} ({member['model']}):")
    print(f"  Grade: {member.get('grade', 'N/A')}")
    print(f"  Top Concern: {member.get('top_concern', 'N/A')}")

# Run with an agenda vote
result = run_board(briefing, agenda="Should we remove the Funding Squeeze setup?")
for member in result.get("responses", []):
    print(f"  {member['role']}: {member.get('vote', 'N/A')} — {member.get('reasoning', '')[:80]}")
