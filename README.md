# AI Board of Directors

Multi-model AI consensus system with role-specialized expert agents. Get orthogonal perspectives on decisions by having different AI models evaluate the same data from distinct professional viewpoints.

## The Pattern

Instead of asking one AI model for an answer, assemble a "board" of AI models — each with a different expert role — and synthesize their collective judgment:

| Role | Model | Perspective |
|------|-------|-------------|
| Risk Officer | Claude Opus | Conservative, downside-focused |
| Quant Analyst | Claude Sonnet (Thinking) | Data-driven, statistical |
| Infra Engineer | GPT-5.4 | Systems reliability, technical debt |
| Devil's Advocate | Grok | Contrarian, challenges assumptions |
| Strategist | Gemini 3.1 Pro | Long-term vision, market positioning |
| Fund Manager | GPT-5.4 Medium | P&L focused, practical |

## Why This Works

- **Role diversity produces better decisions** — a Risk Officer's "no" on leverage might save you from a blowup
- **Disagreement is informative** — if 5 models vote YES and 1 votes NO, the NO reasoning is valuable
- **Transparent consensus** — you get both the majority vote AND all reasoning
- **Cheap experiments** — run board meetings on major decisions before committing

## Usage

### Board Meeting (Grade + Assess)

```python
from ai_board import run_board

result = run_board("Your briefing here...")
print(f"Grade: {result['consensus_grade']}, Agreement: {result['agreement_pct']}%")
```

### Agenda Vote (Yes/No Decision)

```python
result = run_board(briefing, agenda="Should we proceed with X?")
for r in result["responses"]:
    print(f"{r['role']}: {r['vote']} — {r['reasoning']}")
```

### Market Pulse (3-Model Consensus)

```python
from market_pulse import run_pulse

result = run_pulse("BTC at $70K, F&G: 65, Funding: 0.01%...")
print(f"Consensus: {result['consensus']}, Agreement: {result['agreement_pct']}%")
```

## Architecture

```
Briefing ──→ Board Members (parallel)
              ├── Risk Officer ──→ [Grade, Confidence, Concern, Recommendation]
              ├── Quant Analyst ──→ [Grade, Confidence, Concern, Recommendation]
              ├── Infra Engineer ──→ [Grade, Confidence, Concern, Recommendation]
              ├── Devil's Advocate ──→ [Grade, Confidence, Concern, Recommendation]
              ├── Strategist ──→ [Grade, Confidence, Concern, Recommendation]
              └── Fund Manager ──→ [Grade, Confidence, Concern, Recommendation]
                                    ↓
                           Consensus Computation
                           (avg confidence, concern list, grade distribution)
                                    ↓
                           Board Meeting Record
                           (persisted to board_history.json)
```

## Structured Output Extraction

Each model's response is parsed with regex to extract:
- **GRADE**: A+ through F
- **CONFIDENCE**: 0-100%
- **TOP_CONCERN**: One-line summary
- **RECOMMENDATION**: Action item
- **VOTE** (agenda mode): YES/NO
- **REASONING** (agenda mode): Explanation

This works reliably across model families (Claude, GPT, Gemini, Grok) without requiring JSON mode or function calling.

## Requirements

- Python 3.11+
- `requests`
- Access to OpenAI-compatible API endpoints (proxied or direct)

## Applications Beyond Trading

- **Code review**: Each model reviews from different angles (security, performance, maintainability)
- **Product decisions**: PM, Engineer, Designer, Customer Success perspectives
- **Investment committee**: Bull case, bear case, risk assessment
- **Hiring**: Different interviewers evaluate same candidate
- **Contract review**: Legal, business, technical perspectives
