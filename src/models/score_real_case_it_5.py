from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_models import TARGET, get_feature_matrix, temporal_split
from src.utils.paths import MODELING_DIR, MODEL_REPORTS_DIR, MODELS_DIR, ensure_dir

DATASET = MODELING_DIR / "churn_modeling_dataset_it_3c_top75.csv"
MODEL = MODELS_DIR / "best_model_it_3c_top75.joblib"
TEST_SCORES = MODEL_REPORTS_DIR / "test_scores_best_model_it_3c_top75.csv"
OUTPUT_JSON = MODEL_REPORTS_DIR / "real_case_score_it_5.json"

TOP_K_THRESHOLDS = {
    "top_1pct": 0.01,
    "top_5pct": 0.05,
    "top_10pct": 0.10,
    "top_20pct": 0.20,
}

KEY_FEATURES = [
    "soporte_canal_principal",
    "soporte_motivo_principal",
    "fact_importe_total",
    "fact_importe_total_lag_1m",
    "fact_dias_retraso_pago_roll_3m",
    "fact_impago_flag_roll_3m",
    "red_indice_calidad_global_lag_2m",
    "red_cobertura_5g_pct_lag_2m",
    "fact_stress_calidad_lag_roll_3m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score de un caso real con el modelo it_3c_top75.")
    parser.add_argument("--cliente-id", default=None, help="Cliente a puntuar, por ejemplo C000123.")
    parser.add_argument("--fecha", default=None, help="Mes a puntuar en formato YYYY-MM-01.")
    parser.add_argument(
        "--mode",
        choices=["highest_score", "random_churn", "random_non_churn"],
        default="highest_score",
        help="Caso real a elegir si no se informa cliente-id/fecha.",
    )
    return parser.parse_args()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    if not DATASET.exists():
        raise FileNotFoundError(f"No existe {DATASET}.")
    if not MODEL.exists():
        raise FileNotFoundError(f"No existe {MODEL}.")
    if not TEST_SCORES.exists():
        raise FileNotFoundError(f"No existe {TEST_SCORES}.")

    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    scores = pd.read_csv(TEST_SCORES, parse_dates=["fecha"])
    model = joblib.load(MODEL)
    return df, scores, model


def choose_case(df: pd.DataFrame, scores: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    if args.cliente_id and args.fecha:
        fecha = pd.to_datetime(args.fecha)
        match = df[(df["cliente_id"] == args.cliente_id) & (df["fecha"] == fecha)]
        if match.empty:
            raise ValueError(f"No existe caso para cliente_id={args.cliente_id}, fecha={args.fecha}.")
        return match.iloc[0]

    if args.mode == "highest_score":
        selected = scores.sort_values("score_churn", ascending=False).iloc[0]
    elif args.mode == "random_churn":
        selected = scores[scores["y_real"] == 1].sample(1, random_state=42).iloc[0]
    else:
        selected = scores[scores["y_real"] == 0].sample(1, random_state=42).iloc[0]

    match = df[(df["cliente_id"] == selected["cliente_id"]) & (df["fecha"] == selected["fecha"])]
    if match.empty:
        raise ValueError("El caso elegido desde scores no existe en el dataset de modelado.")
    return match.iloc[0]


def risk_position(score: float, scores: pd.DataFrame) -> dict[str, object]:
    rank = int((scores["score_churn"] > score).sum() + 1)
    total = int(len(scores))
    percentile_risk = 1 - ((rank - 1) / total)
    top_membership = {
        name: rank <= int(round(total * frac))
        for name, frac in TOP_K_THRESHOLDS.items()
    }
    return {
        "rank_by_risk": rank,
        "total_test_cases": total,
        "risk_percentile": percentile_risk,
        "top_membership": top_membership,
    }


def main() -> None:
    ensure_dir(MODEL_REPORTS_DIR)
    args = parse_args()
    df, scores, model = load_inputs()
    case = choose_case(df, scores, args)

    case_df = pd.DataFrame([case])
    X_case, y_case = get_feature_matrix(case_df)
    score = float(model.predict_proba(X_case)[:, 1][0])
    pred_05 = int(score >= 0.5)
    position = risk_position(score, scores)

    feature_snapshot = {
        col: (None if pd.isna(case[col]) else case[col].item() if hasattr(case[col], "item") else case[col])
        for col in KEY_FEATURES
        if col in case.index
    }

    output = {
        "cliente_id": case["cliente_id"],
        "fecha": str(pd.to_datetime(case["fecha"]).date()),
        "target_real_churn_t_plus_1": int(y_case.iloc[0]),
        "score_churn": score,
        "pred_churn_threshold_0_5": pred_05,
        "operational_recommendation": (
            "contactar" if position["top_membership"]["top_10pct"] else "no_prioritario"
        ),
        "risk_position": position,
        "feature_snapshot": feature_snapshot,
        "interpretation": [
            "score_churn es la probabilidad estimada/ranking score de churn para t+1.",
            "La recomendacion operativa usa pertenencia al Top 10% de riesgo, no threshold 0.5.",
            "target_real_churn_t_plus_1 solo existe en test historico; en produccion no estaria disponible.",
        ],
    }
    OUTPUT_JSON.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"\nGuardado en: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
