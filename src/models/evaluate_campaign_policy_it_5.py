from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import MODEL_REPORTS_DIR, ensure_dir, project_path

INPUT_SCORES = MODEL_REPORTS_DIR / "test_scores_best_model_it_3c_top75.csv"
POLICY_REPORT = MODEL_REPORTS_DIR / "campaign_policy_it_5.csv"
COST_BENEFIT_REPORT = MODEL_REPORTS_DIR / "cost_benefit_scenarios_it_5.csv"
RECOMMENDATION_JSON = MODEL_REPORTS_DIR / "operational_recommendation_it_5.json"

TOP_K_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

## Supuestos configurables de negocio.
## Estos valores no son "verdad"; son escenarios para entender sensibilidad.
CONTACT_COSTS = [1.0, 3.0, 5.0]
RETENTION_VALUES = [50.0, 100.0, 200.0]
SAVE_RATES = [0.05, 0.10, 0.20]


def load_scores() -> pd.DataFrame:
    if not INPUT_SCORES.exists():
        raise FileNotFoundError(f"No existe {INPUT_SCORES}. Ejecuta primero Iteracion 3C.")
    df = pd.read_csv(INPUT_SCORES, parse_dates=["fecha"])
    required = {"cliente_id", "fecha", "y_real", "score_churn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    return df


def build_campaign_policy(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values("score_churn", ascending=False).reset_index(drop=True)
    total_rows = len(ranked)
    total_churners = int(ranked["y_real"].sum())
    baseline_rate = total_churners / total_rows

    rows = []
    for frac in TOP_K_FRACTIONS:
        contacted = max(1, int(round(total_rows * frac)))
        top = ranked.head(contacted)
        captured = int(top["y_real"].sum())
        precision = captured / contacted
        capture_rate = captured / total_churners if total_churners else np.nan
        false_positives = contacted - captured
        rows.append(
            {
                "policy": f"top_{int(frac * 100)}pct",
                "top_fraction": frac,
                "contacted": contacted,
                "captured_churners": captured,
                "false_positives": false_positives,
                "precision_at_k": precision,
                "capture_rate": capture_rate,
                "baseline_churn_rate": baseline_rate,
                "lift": precision / baseline_rate if baseline_rate else np.nan,
                "min_score": float(top["score_churn"].min()),
                "max_score": float(top["score_churn"].max()),
            }
        )
    return pd.DataFrame(rows)


def build_cost_benefit(policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in policy.iterrows():
        for contact_cost in CONTACT_COSTS:
            for retention_value in RETENTION_VALUES:
                for save_rate in SAVE_RATES:
                    expected_saved = row["captured_churners"] * save_rate
                    campaign_cost = row["contacted"] * contact_cost
                    expected_value = expected_saved * retention_value
                    net_value = expected_value - campaign_cost
                    roi = net_value / campaign_cost if campaign_cost else np.nan
                    break_even_save_rate = (
                        campaign_cost / (row["captured_churners"] * retention_value)
                        if row["captured_churners"] and retention_value
                        else np.nan
                    )
                    rows.append(
                        {
                            "policy": row["policy"],
                            "contacted": int(row["contacted"]),
                            "captured_churners": int(row["captured_churners"]),
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
                            "roi": roi,
                            "break_even_save_rate": break_even_save_rate,
                        }
                    )
    return pd.DataFrame(rows)


def choose_recommendations(policy: pd.DataFrame, cost_benefit: pd.DataFrame) -> dict[str, object]:
    best_net = cost_benefit.sort_values(["net_value", "roi"], ascending=False).iloc[0].to_dict()
    best_roi_positive = cost_benefit[cost_benefit["net_value"] > 0]
    best_roi = (
        best_roi_positive.sort_values(["roi", "net_value"], ascending=False).iloc[0].to_dict()
        if not best_roi_positive.empty
        else None
    )
    top5 = policy[policy["policy"] == "top_5pct"].iloc[0].to_dict()
    top10 = policy[policy["policy"] == "top_10pct"].iloc[0].to_dict()
    top20 = policy[policy["policy"] == "top_20pct"].iloc[0].to_dict()

    return {
        "iteration": "it_5",
        "source_scores": project_path(INPUT_SCORES),
        "objective": "Definir politicas operativas Top-K y escenarios de coste-beneficio.",
        "recommended_default_policy": "top_10pct",
        "recommended_default_policy_reason": (
            "Top 10% captura una proporcion relevante de churners sin llegar al volumen alto de Top 20%."
        ),
        "top_5pct": top5,
        "top_10pct": top10,
        "top_20pct": top20,
        "best_net_value_scenario": best_net,
        "best_roi_positive_scenario": best_roi,
        "business_notes": [
            "Top-K es preferible a threshold fijo cuando la capacidad comercial es limitada.",
            "Los escenarios de coste-beneficio son sensibles a contact_cost, retention_value y save_rate.",
            "Antes de desplegar, fijar supuestos economicos reales con negocio.",
            "Comparar tambien saturacion comercial y riesgo reputacional de contactar falsos positivos.",
        ],
    }


def run_campaign_policy() -> dict[str, Path]:
    ensure_dir(MODEL_REPORTS_DIR)
    df = load_scores()
    policy = build_campaign_policy(df)
    cost_benefit = build_cost_benefit(policy)
    recommendation = choose_recommendations(policy, cost_benefit)

    policy.to_csv(POLICY_REPORT, index=False)
    cost_benefit.to_csv(COST_BENEFIT_REPORT, index=False)
    RECOMMENDATION_JSON.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    return {
        "policy_report": POLICY_REPORT,
        "cost_benefit_report": COST_BENEFIT_REPORT,
        "recommendation": RECOMMENDATION_JSON,
    }


if __name__ == "__main__":
    for name, path in run_campaign_policy().items():
        print(f"{name}: {path}")
