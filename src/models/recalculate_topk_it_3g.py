from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCORES_PATH = ROOT / "reports" / "models" / "test_scores_best_model_it_3g_logistic_semantic_imputation.csv"
OUTPUT_TOPK = ROOT / "reports" / "models" / "lift_topk_it_3g_logistic_semantic_imputation.csv"
OUTPUT_COST_BENEFIT = ROOT / "reports" / "models" / "cost_benefit_scenarios_it_3g_logistic_semantic_imputation.csv"

TOP_K_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
CONTACT_COSTS = [1.0, 3.0, 5.0]
RETENTION_VALUES = [50.0, 100.0, 200.0]
SAVE_RATES = [0.05, 0.10, 0.20]


def build_topk(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values("score_churn", ascending=False).reset_index(drop=True)
    total_rows = len(ranked)
    total_churners = int(ranked["y_real"].sum())
    baseline_rate = total_churners / total_rows

    rows = []
    for frac in TOP_K_FRACTIONS:
        contacted = max(1, int(round(total_rows * frac)))
        top = ranked.head(contacted)
        captured = int(top["y_real"].sum())
        precision_at_k = captured / contacted
        rows.append(
            {
                "policy": f"top_{int(frac * 100)}pct",
                "top_fraction": frac,
                "contacted": contacted,
                "captured_churners": captured,
                "false_positives": contacted - captured,
                "precision_at_k": precision_at_k,
                "capture_rate": captured / total_churners,
                "baseline_churn_rate": baseline_rate,
                "lift": precision_at_k / baseline_rate,
                "min_score": float(top["score_churn"].min()),
                "max_score": float(top["score_churn"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_cost_benefit(policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in policy.iterrows():
        for contact_cost, retention_value, save_rate in itertools.product(
            CONTACT_COSTS, RETENTION_VALUES, SAVE_RATES
        ):
            captured = row["captured_churners"]
            contacted = row["contacted"]
            expected_saved = captured * save_rate
            campaign_cost = contacted * contact_cost
            expected_value = expected_saved * retention_value
            net_value = expected_value - campaign_cost
            rows.append(
                {
                    "policy": row["policy"],
                    "contacted": int(contacted),
                    "captured_churners": int(captured),
                    "precision_at_k": row["precision_at_k"],
                    "capture_rate": row["capture_rate"],
                    "lift": row["lift"],
                    "contact_cost": contact_cost,
                    "retention_value": retention_value,
                    "save_rate": save_rate,
                    "expected_saved_customers": expected_saved,
                    "campaign_cost": campaign_cost,
                    "expected_retention_value": expected_value,
                    "net_value": net_value,
                    "roi": net_value / campaign_cost if campaign_cost else np.nan,
                    "break_even_save_rate": (
                        campaign_cost / (captured * retention_value)
                        if captured and retention_value
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(SCORES_PATH, parse_dates=["fecha"])
    total_rows = len(df)
    total_churners = int(df["y_real"].sum())
    baseline_rate = total_churners / total_rows

    policy = build_topk(df)
    cost_benefit = build_cost_benefit(policy)
    positive = cost_benefit[cost_benefit["net_value"] > 0]

    policy.to_csv(OUTPUT_TOPK, index=False)
    cost_benefit.to_csv(OUTPUT_COST_BENEFIT, index=False)

    print("BASE 3G")
    print(f"rows={total_rows}")
    print(f"total_churners={total_churners}")
    print(f"baseline_churn_rate={baseline_rate:.9f} ({baseline_rate * 100:.4f}%)")
    print()
    print("TOP-K 3G")
    print(policy.to_string(index=False))
    print(f"\nSaved: {OUTPUT_TOPK}")
    print()
    print("COST-BENEFIT 3G")
    print(f"positive_scenarios={len(positive)} of {len(cost_benefit)}")
    print(f"Saved: {OUTPUT_COST_BENEFIT}")
    print()
    print("Best net value")
    print(cost_benefit.sort_values(["net_value", "roi"], ascending=False).head(3).to_string(index=False))
    print()
    print("Best positive ROI")
    if positive.empty:
        print("No positive scenarios")
    else:
        print(positive.sort_values(["roi", "net_value"], ascending=False).head(3).to_string(index=False))


if __name__ == "__main__":
    main()
