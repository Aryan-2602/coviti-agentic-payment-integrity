"""
Streamlit dashboard for the Payment Integrity Multi-Agent POC.
Run with: streamlit run app.py
"""
import time
import datetime
import streamlit as st
import pandas as pd

from generate_data import generate_claims, PROCEDURE_CODES
from agents import run_pipeline, Classifier, ReasoningAgent

st.set_page_config(page_title="Payment Integrity Multi-Agent Demo", layout="wide")

REQUIRED_COLUMNS = ["provider_id", "patient_id", "procedure_code", "procedure_desc",
                    "claim_date", "billing_amount"]

# ---------------------------------------------------------------- session state init

if "data_ready" not in st.session_state:
    st.session_state.data_ready = False


# ---------------------------------------------------------------- landing screen

if not st.session_state.data_ready:
    st.title("🩺 Agentic Claims Investigation Copilot")
    st.caption(
        "A 3-agent pipeline for healthcare payment integrity — anomaly detection → "
        "classification → LLM reasoning agent. All synthetic-data runs use no real "
        "patient or provider data."
    )
    st.divider()

    st.subheader("Choose a dataset to get started")

    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown("#### Option A — Demo dataset")
        st.write("Load a pre-built synthetic dataset of **604 claims** across 25 providers, "
                 "with injected upcoding, duplicate billing, and unbundling patterns.")
        if st.button("▶ Use demo dataset", use_container_width=True, type="primary"):
            st.session_state.raw_df = generate_claims()
            st.session_state.data_ready = True
            st.rerun()

    with right_col:
        st.markdown("#### Option B — Upload your own CSV")
        st.write("Upload a claims file. Required columns:")
        st.code(", ".join(REQUIRED_COLUMNS), language=None)

        uploaded = st.file_uploader("Upload claims CSV", type=["csv"],
                                    label_visibility="collapsed")
        if uploaded is not None:
            try:
                user_df = pd.read_csv(uploaded, dtype={"procedure_code": str})
            except Exception as e:
                st.error(f"Could not parse CSV: {e}")
                user_df = None

            if user_df is not None:
                missing = [c for c in REQUIRED_COLUMNS if c not in user_df.columns]
                if missing:
                    st.error(f"Missing required column(s): **{', '.join(missing)}**. "
                             "Please fix your file and re-upload.")
                else:
                    # auto-generate claim_id if absent so the dashboard works
                    if "claim_id" not in user_df.columns:
                        user_df.insert(0, "claim_id",
                                       [f"CLM-{i+1:05d}" for i in range(len(user_df))])
                    st.success(f"File looks good — {len(user_df):,} rows detected.")
                    st.dataframe(user_df.head(5), use_container_width=True, hide_index=True)
                    if st.button("✅ Confirm and load this dataset",
                                 use_container_width=True, type="primary"):
                        st.session_state.raw_df = user_df
                        st.session_state.data_ready = True
                        st.rerun()

    st.stop()


# ---------------------------------------------------------------- pipeline setup (once per dataset)

@st.cache_resource
def get_reasoning_agent():
    return ReasoningAgent()


if "scored_df" not in st.session_state:
    with st.spinner("Fitting anomaly detection model on historical claims..."):
        scored_df, detector = run_pipeline(st.session_state.raw_df)
    st.session_state.scored_df = scored_df
    st.session_state.detector = detector

df = st.session_state.scored_df
detector = st.session_state.detector
agent3 = get_reasoning_agent()

flagged = df[df["is_flagged"]].sort_values("anomaly_score", ascending=False)


# ---------------------------------------------------------------- persistent header

header_left, header_right = st.columns([5, 1])
with header_left:
    st.title("🩺 Agentic Claims Investigation Copilot")
    st.caption(
        "A 3-agent pipeline for healthcare payment integrity — anomaly detection → "
        "classification → LLM reasoning agent. All data is synthetic; no real patient or provider data is used."
    )
with header_right:
    st.write("")  # vertical nudge
    st.write("")
    if st.button("↺ Change dataset", use_container_width=True):
        for key in ("data_ready", "raw_df", "scored_df", "detector"):
            st.session_state.pop(key, None)
        st.rerun()

if agent3.is_live:
    st.success("🟢 **Live mode** — Agent 3 is calling OpenAI directly for reasoning and follow-up Q&A.")
else:
    st.warning(
        "📄 **Template mode** — no `OPENAI_API_KEY` detected in your environment. "
        "Agent 3 is using deterministic templates instead of a live model call. "
        "Set the env var and restart `streamlit run app.py` to enable live reasoning + chat."
    )


# ---------------------------------------------------------------- shared claim detail + chat UI

def render_claim_detail(claim: dict, key_prefix: str):
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Agent 1 — Anomaly Detector**")
        st.write("🚩 Flagged" if claim["is_flagged"] else "✅ Not flagged")
        st.write(f"Anomaly score: `{claim['anomaly_score']:.3f}`")
        st.write(f"Ratio to peer median billing for this procedure: `{claim['amount_ratio_to_peer_median']:.2f}x`")
        st.write(f"Same-day duplicate count: `{claim['same_day_dup_count']}`")
        st.markdown("**Agent 2 — Classifier**")
        st.write(f"Predicted category: `{claim['predicted_category']}`")
    with c2:
        st.markdown("**Agent 3 — Reasoning Agent** (investigation summary)")
        st.info(claim.get("investigation_summary", "Not yet reviewed."))

    st.markdown("**💬 Ask Agent 3 a follow-up question about this claim**")
    hist_key = f"chat_{key_prefix}"
    if hist_key not in st.session_state:
        st.session_state[hist_key] = []

    for msg in st.session_state[hist_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    q_col, btn_col = st.columns([5, 1])
    question = q_col.text_input(
        "Follow-up question", key=f"qbox_{key_prefix}",
        placeholder="e.g. What documentation should I request?", label_visibility="collapsed",
    )
    ask_clicked = btn_col.button("Ask", key=f"ask_{key_prefix}", use_container_width=True)

    if ask_clicked and question.strip():
        st.session_state[hist_key].append({"role": "user", "content": question})
        with st.spinner("Agent 3 thinking..."):
            answer = agent3.answer_followup(claim, question, history=st.session_state[hist_key][:-1])
        st.session_state[hist_key].append({"role": "assistant", "content": answer})
        st.rerun()


# ---------------------------------------------------------------- tabs

tab1, tab2 = st.tabs(["📊 Claims Portfolio", "🆕 Submit a New Claim"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total claims processed", len(df))
    col2.metric("Flagged for investigation", len(flagged))
    col3.metric("Flag rate", f"{len(flagged) / len(df) * 100:.1f}%")

    left, right = st.columns([2, 1])
    with left:
        st.subheader("🚩 Flagged Claims")
        st.dataframe(
            flagged[["claim_id", "provider_id", "procedure_code", "billing_amount",
                     "amount_ratio_to_peer_median", "predicted_category", "anomaly_score"]]
            .rename(columns={"amount_ratio_to_peer_median": "ratio_to_peer_median"}),
            use_container_width=True, hide_index=True, height=260,
        )
    with right:
        st.subheader("Flags by category")
        st.bar_chart(flagged["predicted_category"].value_counts())

    st.subheader("🔎 Investigate a Flagged Claim")
    selected_id = st.selectbox("Select a claim to see the agent reasoning trail:", flagged["claim_id"])
    claim = df[df["claim_id"] == selected_id].iloc[0].to_dict()
    render_claim_detail(claim, key_prefix=f"hist_{selected_id}")

with tab2:
    st.subheader("Submit a claim and watch the agents investigate it live")
    st.caption("This runs the same fitted Agent 1 model and the same Agent 2 / Agent 3 logic as the portfolio tab — nothing here is precomputed.")

    presets = {
        "Plain claim": dict(provider_id="PRV-1003", procedure_code="99213", billing_amount=90.0, same_day_repeat=1),
        "🚩 Looks like upcoding": dict(provider_id="PRV-1003", procedure_code="99214", billing_amount=340.0, same_day_repeat=1),
        "🚩 Looks like duplicate billing": dict(provider_id="PRV-1003", procedure_code="80053", billing_amount=35.0, same_day_repeat=3),
        "🚩 Looks like unbundling": dict(provider_id="PRV-1003", procedure_code="80053", billing_amount=95.0, same_day_repeat=1),
    }
    preset_cols = st.columns(len(presets))
    for col, (label, vals) in zip(preset_cols, presets.items()):
        if col.button(label, use_container_width=True):
            st.session_state["preset_vals"] = vals
            st.rerun()

    pv = st.session_state.get("preset_vals", presets["Plain claim"])
    provider_options = sorted(df["provider_id"].unique().tolist()) + ["NEW-PROVIDER-001"]

    with st.form("new_claim_form"):
        c1, c2 = st.columns(2)
        provider_id = c1.selectbox(
            "Provider", provider_options,
            index=provider_options.index(pv["provider_id"]) if pv["provider_id"] in provider_options else 0,
        )
        patient_id = c2.text_input("Patient ID", value="PT-9001")
        procedure_code = c1.selectbox(
            "Procedure code", list(PROCEDURE_CODES.keys()),
            index=list(PROCEDURE_CODES.keys()).index(pv["procedure_code"]),
            format_func=lambda code: f"{code} — {PROCEDURE_CODES[code][0]}",
        )
        billing_amount = c2.number_input("Billing amount ($)", min_value=0.0, value=float(pv["billing_amount"]), step=5.0)
        claim_date = c1.date_input("Claim date", value=datetime.date(2026, 5, 1))
        same_day_repeat = c2.number_input(
            "Times billed for this patient/procedure/date (incl. this one)",
            min_value=1, value=int(pv["same_day_repeat"]), step=1,
            help="Set to 2+ to simulate duplicate billing for the same patient, procedure, and date.",
        )
        submitted = st.form_submit_button("🚀 Run investigation", use_container_width=True)

    if submitted:
        new_id = f"CLM-LIVE-{int(time.time())}"
        base_claim = {
            "claim_id": new_id,
            "provider_id": provider_id,
            "patient_id": patient_id,
            "procedure_code": procedure_code,
            "procedure_desc": PROCEDURE_CODES[procedure_code][0],
            "claim_date": str(claim_date),
            "billing_amount": billing_amount,
        }

        with st.status("Running multi-agent investigation...", expanded=True) as status:
            st.write("🔍 **Agent 1 — Anomaly Detector**: scoring against the peer billing rate for this procedure...")
            time.sleep(0.5)
            if same_day_repeat > 1:
                extra_rows = pd.DataFrame([{
                    "provider_id": provider_id, "patient_id": patient_id,
                    "claim_date": str(claim_date), "procedure_code": procedure_code,
                }] * (same_day_repeat - 1))
                detector._history = pd.concat([detector._history, extra_rows], ignore_index=True)
            scored = detector.score_one(base_claim)
            st.write(
                f"&nbsp;&nbsp;→ anomaly score `{scored['anomaly_score']:.2f}` · "
                f"ratio to peer median `{scored['amount_ratio_to_peer_median']:.2f}x` · "
                f"same-day count `{scored['same_day_dup_count']}`"
            )

            st.write("🏷️ **Agent 2 — Classifier**: triaging why it was (or wasn't) flagged...")
            time.sleep(0.5)
            scored["predicted_category"] = Classifier.classify(scored)
            st.write(f"&nbsp;&nbsp;→ category: `{scored['predicted_category']}`")

            mode_note = "live OpenAI call" if agent3.is_live else "template fallback — set OPENAI_API_KEY for a live call"
            st.write(f"🧠 **Agent 3 — Reasoning Agent**: drafting investigation summary ({mode_note})...")
            time.sleep(0.5)
            scored["investigation_summary"] = agent3.run(scored)

            status.update(label="✅ Investigation complete", state="complete", expanded=True)

        st.session_state["last_live_claim"] = scored
        st.session_state.pop(f"chat_live_{new_id}", None)

    if "last_live_claim" in st.session_state:
        st.divider()
        st.subheader("Result")
        live_claim = st.session_state["last_live_claim"]
        render_claim_detail(live_claim, key_prefix=f"live_{live_claim['claim_id']}")

st.divider()
st.caption(
    "POC built for Cotiviti Intern Assessment — Option 2: Clinical Decision Making & "
    "Pattern Recognition in Health Care (Agentic Generative AI for Payment Integrity, TPO)."
)
