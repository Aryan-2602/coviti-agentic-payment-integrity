"""
Generates synthetic healthcare claims data for the Payment Integrity
Multi-Agent Demo. No real patient or provider data is used anywhere
in this POC -- this sidesteps PHI/compliance concerns entirely, which
is intentional and is referenced in the written report.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

PROCEDURE_CODES = {
    "99213": ("Office visit, established patient, low complexity", 90),
    "99214": ("Office visit, established patient, moderate complexity", 130),
    "99215": ("Office visit, established patient, high complexity", 185),
    "93000": ("Electrocardiogram, routine", 40),
    "80053": ("Comprehensive metabolic panel", 35),
    "71046": ("Chest X-ray, 2 views", 65),
    "20610": ("Joint injection, major joint", 110),
}

N_PROVIDERS = 25
N_CLAIMS = 600

def generate_claims():
    rows = []
    provider_ids = [f"PRV-{1000+i}" for i in range(N_PROVIDERS)]

    # assign each provider a "normal" baseline billing behavior
    provider_baseline = {
        p: np.random.uniform(0.85, 1.15) for p in provider_ids
    }

    # pick a handful of providers to inject anomalous behavior into
    anomalous_providers = np.random.choice(provider_ids, size=4, replace=False)
    anomaly_types = {
        anomalous_providers[0]: "upcoding",
        anomalous_providers[1]: "duplicate_billing",
        anomalous_providers[2]: "unbundling",
        anomalous_providers[3]: "volume_spike",
    }

    start_date = datetime(2026, 1, 1)
    claim_id = 1

    for _ in range(N_CLAIMS):
        provider = np.random.choice(provider_ids)
        code = np.random.choice(list(PROCEDURE_CODES.keys()))
        desc, base_price = PROCEDURE_CODES[code]
        claim_date = start_date + timedelta(days=int(np.random.uniform(0, 150)))
        patient_id = f"PT-{np.random.randint(1, 400):04d}"

        amount = base_price * provider_baseline[provider] * np.random.uniform(0.95, 1.05)
        anomaly_flag_truth = "normal"

        if provider in anomaly_types:
            kind = anomaly_types[provider]
            if kind == "upcoding" and code in ("99213", "99214"):
                # bills low-complexity visits as if high-complexity
                amount = PROCEDURE_CODES["99215"][1] * np.random.uniform(1.05, 1.25)
                anomaly_flag_truth = "upcoding"
            elif kind == "duplicate_billing" and np.random.rand() < 0.35:
                rows.append({
                    "claim_id": f"CLM-{claim_id:05d}",
                    "provider_id": provider,
                    "patient_id": patient_id,
                    "procedure_code": code,
                    "procedure_desc": desc,
                    "claim_date": claim_date.strftime("%Y-%m-%d"),
                    "billing_amount": round(amount, 2),
                    "anomaly_flag_truth": "duplicate_billing",
                })
                claim_id += 1
                amount = amount  # the duplicate, same patient/code/day
                anomaly_flag_truth = "duplicate_billing"
            elif kind == "unbundling" and code in ("80053",):
                amount = amount * np.random.uniform(2.2, 2.8)  # billed as separate components
                anomaly_flag_truth = "unbundling"
            elif kind == "volume_spike":
                amount = amount * np.random.uniform(0.95, 1.05)
                anomaly_flag_truth = "volume_spike"

        rows.append({
            "claim_id": f"CLM-{claim_id:05d}",
            "provider_id": provider,
            "patient_id": patient_id,
            "procedure_code": code,
            "procedure_desc": desc,
            "claim_date": claim_date.strftime("%Y-%m-%d"),
            "billing_amount": round(amount, 2),
            "anomaly_flag_truth": anomaly_flag_truth,
        })
        claim_id += 1

    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = generate_claims()
    df.to_csv("claims_data.csv", index=False)
    print(f"Generated {len(df)} synthetic claims across {N_PROVIDERS} providers.")
    print(df["anomaly_flag_truth"].value_counts())
