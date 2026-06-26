# Agentic Payment Integrity — Cotiviti Intern Assessment POC

**Option 2:** Clinical Decision Making and Pattern Recognition in Health Care
(Agentic Generative AI, Classification, Anomaly Detection for Treatment,
Payment, & Operations)

## What this is

A 3-agent pipeline showing how agentic GenAI could support payment integrity
and claims investigation workflows:

1. **Agent 1 — Anomaly Detector**: Isolation Forest flags claims priced well
   outside the *peer* norm for that exact procedure code (compared network-wide,
   not against a provider's own history — a provider's own median would be
   self-contaminated if that provider is inflating prices), or billed multiple
   times for the same patient/procedure/day.
2. **Agent 2 — Classifier**: rule-based triage labels *why* a claim was
   flagged (duplicate billing, unbundling, upcoding, volume outlier).
3. **Agent 3 — Reasoning Agent**: an LLM agent (OpenAI) chains the evidence
   from Agents 1 and 2 into a written investigation summary and recommendation,
   and can field free-form follow-up questions about any claim it reviewed.
   Summaries are generated lazily — only when a claim is actually viewed —
   and cached so the same claim is never scored twice.

All claims data is **synthetic** — generated in `cotiviti_poc/generate_data.py`
— so there is no PHI or real provider data anywhere in this repo.

## Running the app

```bash
cd cotiviti_poc
pip install -r requirements.txt
streamlit run app.py
```

The landing screen lets you either load the built-in 604-claim synthetic
dataset or upload your own CSV (requires columns: `provider_id`, `patient_id`,
`procedure_code`, `procedure_desc`, `claim_date`, `billing_amount`).

Agent 3 uses a live OpenAI API call if `OPENAI_API_KEY` is set:

```bash
export OPENAI_API_KEY=your_key_here
```

Without a key it falls back to deterministic templates automatically, so the
full pipeline runs end-to-end with zero API cost. The app shows a green
**Live mode** or amber **Template mode** badge so it's always clear which
is active.

## Repo layout

```
cotiviti_poc/
  generate_data.py   — synthetic claims data generator
  agents.py          — 3-agent pipeline (anomaly detection, classification, LLM reasoning)
  app.py             — Streamlit dashboard (landing screen, portfolio view, live submission, chat)
  requirements.txt   — dependencies (streamlit, scikit-learn, openai, …)
```
