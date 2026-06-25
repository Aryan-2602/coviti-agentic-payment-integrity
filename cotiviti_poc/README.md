# Agentic Claims Investigation Copilot — POC

**Cotiviti Intern Assessment — Option 2:** Clinical Decision Making and Pattern
Recognition in Health Care (Agentic Generative AI, Classification, Anomaly
Detection for Treatment, Payment, & Operations)

## What this is

A 3-agent pipeline showing how agentic GenAI could support payment
integrity / claims investigation workflows:

1. **Agent 1 — Anomaly Detector**: Isolation Forest flags claims priced
   well outside the *peer* norm for that exact procedure code (compared
   network-wide, not against a provider's own history — a provider's own
   median would be self-contaminated if that provider is the one inflating
   prices), or billed multiple times for the same patient/procedure/day.
2. **Agent 2 — Classifier**: rule-based triage labels *why* a claim was
   flagged (duplicate billing, unbundling, upcoding, volume outlier).
3. **Agent 3 — Reasoning Agent**: an LLM agent chains the evidence from
   Agents 1 and 2 into a written investigation summary and recommendation,
   and can field free-form follow-up questions about any claim it reviewed.

All claims data is **synthetic** — generated in `generate_data.py` — so
there is no PHI or real provider data anywhere in this repo.

## Two ways to use the dashboard

- **📊 Claims Portfolio tab**: browse a pre-scored batch of 604 synthetic
  claims, see which ones got flagged and why, and ask Agent 3 follow-up
  questions about any of them.
- **🆕 Submit a New Claim tab**: enter a brand-new claim (or click one of
  the four quick presets) and watch all three agents score it *live*,
  step by step, against the same fitted model — nothing here is
  precomputed. Follow up with questions about that claim too.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The reasoning agent (Agent 3) will use a live Claude API call — for both
the investigation summary and the follow-up Q&A — if `ANTHROPIC_API_KEY`
is set in your environment:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

If no key is set, it automatically falls back to a deterministic template
for the investigation summary (and says so plainly in the follow-up box),
so the full pipeline still runs end-to-end with zero setup or API cost.
The app shows a green "Live mode" or amber "Template mode" badge at the
top so it's always clear which one is active.

## Files

- `generate_data.py` — synthetic claims data generator
- `agents.py` — the 3-agent pipeline, live single-claim scoring, and follow-up Q&A
- `app.py` — Streamlit dashboard (portfolio view + live submission + chat)
- `claims_data.csv` — example generated dataset
