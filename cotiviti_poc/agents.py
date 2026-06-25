"""
Three-agent pipeline for the Payment Integrity POC.

Agent 1 (AnomalyDetector): unsupervised outlier detection on provider
    billing patterns (Isolation Forest) -- covers the "time-series /
    pattern anomaly detection" piece of the assignment topic.
Agent 2 (Classifier): rule-based triage that labels *why* a claim was
    flagged (upcoding, duplicate billing, unbundling, volume spike) --
    covers "classification."
Agent 3 (ReasoningAgent): an LLM-backed agent that chains the evidence
    from Agents 1 and 2 into a written investigation recommendation, and
    can answer free-form follow-up questions about a claim -- covers
    "chain reasoning" / "agentic generative AI."

Two entry points:
  - run_pipeline(df): batch-scores a whole claims file (used to seed the
    dashboard's "Flagged Claims" table).
  - AnomalyDetector.fit(df) once, then .score_one(claim) + Classifier
    .classify(claim) + ReasoningAgent.run(claim) to score a brand-new,
    user-submitted claim live, against the same fitted model -- this is
    what powers the "Submit a New Claim" tab in app.py.

If OPENAI_API_KEY is not set, Agent 3 falls back to a deterministic
template so the whole pipeline still runs end-to-end without an API key.
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Agent 1: flags claims whose price is unusual relative to the peer norm for that
    procedure code, or that show duplicate same-day billing patterns."""

    def __init__(self, contamination=0.06):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.code_medians = {}      # procedure_code -> peer median billing amount, across ALL providers
        self.overall_median = None  # fallback for an entirely unseen procedure code
        self.threshold = None
        self._history = None  # fitted dataframe, kept for duplicate-count lookups on new claims

    def _reference_price(self, procedure_code) -> float:
        """The fairest baseline for 'is this claim's price normal': the peer median price
        for this exact procedure code across ALL providers in the network. A provider's own
        median is NOT used as the baseline -- if a provider is the one inflating a given
        code, their own historical median for that code is already contaminated by their
        past inflated claims, which silently washes out the anomaly signal. Comparing
        against the network-wide peer price avoids that self-contamination problem and
        mirrors how real payment-integrity outlier detection compares a provider to its peers."""
        if procedure_code in self.code_medians:
            return self.code_medians[procedure_code]
        return self.overall_median

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits the model on a historical batch and returns it scored (used once at startup)."""
        df = df.copy()
        self.code_medians = df.groupby("procedure_code")["billing_amount"].median().to_dict()
        self.overall_median = float(df["billing_amount"].median())

        df["amount_ratio_to_peer_median"] = df.apply(
            lambda r: r["billing_amount"] / self._reference_price(r["procedure_code"]),
            axis=1,
        )
        dup_counts = df.groupby(["provider_id", "patient_id", "claim_date", "procedure_code"]) \
                       .size().reset_index(name="same_day_dup_count")
        df = df.merge(dup_counts, on=["provider_id", "patient_id", "claim_date", "procedure_code"])

        features = df[["amount_ratio_to_peer_median", "same_day_dup_count"]]
        df["anomaly_score"] = -self.model.fit(features).score_samples(features)
        self.threshold = float(np.percentile(df["anomaly_score"], 92))
        df["is_flagged"] = df["anomaly_score"] >= self.threshold

        self._history = df
        return df

    def score_one(self, claim: dict) -> dict:
        """Scores a single new claim live, against the already-fitted model and provider history."""
        claim = dict(claim)
        provider = claim["provider_id"]
        reference = self._reference_price(claim["procedure_code"])
        ratio = claim["billing_amount"] / reference

        hist = self._history
        same_day = hist[
            (hist["provider_id"] == provider)
            & (hist["patient_id"] == claim["patient_id"])
            & (hist["claim_date"] == claim["claim_date"])
            & (hist["procedure_code"] == claim["procedure_code"])
        ]
        dup_count = len(same_day) + 1  # +1 to count the new claim itself

        score = float(-self.model.score_samples(
            pd.DataFrame([[ratio, dup_count]], columns=["amount_ratio_to_peer_median", "same_day_dup_count"])
        )[0])
        flagged = bool(score >= self.threshold)

        claim["amount_ratio_to_peer_median"] = ratio
        claim["same_day_dup_count"] = dup_count
        claim["anomaly_score"] = score
        claim["is_flagged"] = flagged
        claim["is_known_code"] = claim["procedure_code"] in self.code_medians
        return claim


class Classifier:
    """Agent 2: rule-based triage of *why* a flagged claim looks anomalous."""

    @staticmethod
    def classify(claim: dict) -> str:
        if not claim["is_flagged"]:
            return "normal"
        if claim["same_day_dup_count"] > 1:
            return "duplicate_billing"
        if claim["procedure_code"] in ("99213", "99214") and claim["amount_ratio_to_peer_median"] > 1.15:
            return "upcoding_suspected"
        if claim["amount_ratio_to_peer_median"] > 1.8:
            return "unbundling_suspected"
        return "volume_or_pattern_outlier"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch version, used by run_pipeline()."""
        df = df.copy()
        df["predicted_category"] = df.apply(lambda r: self.classify(r.to_dict()), axis=1)
        return df


class ReasoningAgent:
    """Agent 3: chains Agent 1 + Agent 2 evidence into a written recommendation,
    and can field free-form follow-up questions about a claim it has already reviewed."""

    MODEL = "gpt-4o-mini"

    def __init__(self):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError:
                self.client = None

    @property
    def is_live(self) -> bool:
        return self.client is not None

    def _claim_context(self, claim: dict) -> str:
        return (
            f"Claim ID: {claim.get('claim_id', 'N/A')}\n"
            f"Provider: {claim['provider_id']}\n"
            f"Procedure: {claim['procedure_code']} ({claim.get('procedure_desc', 'n/a')})\n"
            f"Billing amount: ${claim['billing_amount']}\n"
            f"Ratio to peer median billing for this procedure code: {claim['amount_ratio_to_peer_median']:.2f}x\n"
            f"Same-day duplicate count (same patient/procedure): {claim['same_day_dup_count']}\n"
            f"Anomaly score: {claim['anomaly_score']:.2f}\n"
            f"Triage category from classifier agent: {claim['predicted_category']}\n"
        )

    def _template_reasoning(self, claim: dict) -> str:
        """Deterministic fallback so the demo runs with zero API cost/key."""
        cat = claim["predicted_category"]
        templates = {
            "duplicate_billing": (
                f"Claim {claim.get('claim_id', 'N/A')} from provider {claim['provider_id']} was billed "
                f"{claim['same_day_dup_count']}x for the same patient, procedure, and date. "
                f"This pattern is consistent with duplicate billing. Recommendation: hold payment "
                f"pending provider confirmation; request itemized documentation for each instance."
            ),
            "unbundling_suspected": (
                f"Claim {claim.get('claim_id', 'N/A')} was billed at {claim['amount_ratio_to_peer_median']:.2f}x "
                f"the peer median price for procedure {claim['procedure_code']}, well above what other "
                f"providers typically charge for this code. This is consistent with unbundling (billing "
                f"component services separately to inflate reimbursement). Recommendation: escalate to a "
                f"coding specialist for a bundling-rules review before payment."
            ),
            "upcoding_suspected": (
                f"Claim {claim.get('claim_id', 'N/A')} bills a routine visit code ({claim['procedure_code']}) at "
                f"{claim['amount_ratio_to_peer_median']:.2f}x the peer median rate for that code, suggesting "
                f"the complexity level may be overstated relative to comparable visits. Recommendation: "
                f"request the clinical note to verify visit complexity matches the billed code."
            ),
            "volume_or_pattern_outlier": (
                f"Claim {claim.get('claim_id', 'N/A')} from provider {claim['provider_id']} falls outside the "
                f"expected billing pattern for this procedure (anomaly score: {claim['anomaly_score']:.2f}) but "
                f"does not match a specific known fraud pattern. Recommendation: route to a human "
                f"reviewer for manual spot-check rather than auto-deny."
            ),
        }
        return templates.get(cat, "No anomaly detected; claim processed normally.")

    def run(self, claim: dict) -> str:
        if not claim["is_flagged"]:
            return "No anomaly detected; claim processed normally."

        if self.client is None:
            return self._template_reasoning(claim)

        prompt = (
            "You are a healthcare payment-integrity investigation assistant. "
            "Given the structured evidence below, write a 2-3 sentence investigation "
            "summary and a clear recommendation (hold payment / escalate / route to human review). "
            "Be specific and cite the numeric evidence.\n\n" + self._claim_context(claim)
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.MODEL, max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content
        except Exception as e:
            return self._template_reasoning(claim) + f"\n\n[Live LLM call failed, used fallback: {e}]"

    def answer_followup(self, claim: dict, question: str, history=None) -> str:
        """Answers a free-form follow-up question about a claim it has already reviewed.
        `history` is a list of {"role": "user"|"assistant", "content": str} from prior turns
        in this conversation, so the agent keeps context across follow-ups."""
        if self.client is None:
            return (
                "Live follow-up Q&A needs an OPENAI_API_KEY in your environment. "
                "With a key set, this question would be sent to the model with the full claim "
                "context and prior conversation, so it can reason about specifics like "
                "what documentation to request or why this pattern triggered a flag."
            )

        system_context = (
            "You are a healthcare payment-integrity investigation assistant. You already "
            "reviewed the claim below and flagged it. Answer the investigator's follow-up "
            "question directly and specifically, grounded in the evidence given. Keep answers "
            "to 2-4 sentences unless more detail is clearly needed.\n\n" + self._claim_context(claim)
        )
        messages = [{"role": "system", "content": system_context}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": question})

        try:
            resp = self.client.chat.completions.create(model=self.MODEL, max_tokens=300, messages=messages)
            return resp.choices[0].message.content
        except Exception as e:
            return f"Live call failed: {e}"


def run_pipeline(df: pd.DataFrame):
    """Orchestrator: runs Agent 1 and Agent 2 over a batch.
    Agent 3 is called lazily from the UI when a specific claim is selected.
    Returns (scored_df, fitted_detector) -- the fitted detector is reused to score new,
    user-submitted claims live in the 'Submit a New Claim' tab."""
    detector = AnomalyDetector()
    df = detector.fit(df)
    df = Classifier().run(df)
    return df, detector


if __name__ == "__main__":
    raw = pd.read_csv("claims_data.csv", dtype={"procedure_code": str})
    result, _ = run_pipeline(raw)
    flagged = result[result["is_flagged"]]
    print(f"Flagged {len(flagged)} of {len(result)} claims.\n")
    for _, row in flagged.head(3).iterrows():
        print(f"--- {row['claim_id']} ({row['predicted_category']}) ---")
