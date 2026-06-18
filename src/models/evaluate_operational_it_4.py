from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import MODEL_REPORTS_DIR, ensure_dir, project_path

INPUT_SCORES = MODEL_REPORTS_DIR / "test_scores_best_model_it_3c_top75.csv"
THRESHOLD_REPORT = MODEL_REPORTS_DIR / "threshold_optimization_it_4.csv"
LIFT_REPORT = MODEL_REPORTS_DIR / "lift_topk_it_4.csv"
SUMMARY_JSON = MODEL_REPORTS_DIR / "operational_summary_it_4.json"

THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)
TOP_K_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def load_scores() -> pd.DataFrame:
    if not INPUT_SCORES.exists():
        raise FileNotFoundError(f"No existe {INPUT_SCORES}. Ejecuta primero Iteracion 3C.")

    df = pd.read_csv(INPUT_SCORES, parse_dates=["fecha"])
    required = {"cliente_id", "fecha", "y_real", "score_churn"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
    return df


def confusion_counts(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def build_threshold_report(df: pd.DataFrame) -> pd.DataFrame:
    y_true = df["y_real"].astype(int)
    total_rows = len(df)
    total_churners = int(y_true.sum())

    rows = []
    for threshold in THRESHOLDS:
        y_pred = (df["score_churn"] >= threshold).astype(int).to_numpy()
        counts = confusion_counts(y_true, y_pred)
        contacted = counts["tp"] + counts["fp"]
        rows.append(
            {
                "threshold": threshold,
                "contacted": contacted,
                "contacted_rate": contacted / total_rows,
                "captured_churners": counts["tp"],
                "capture_rate": counts["tp"] / total_churners if total_churners else np.nan,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                **counts,
            }
        )
    return pd.DataFrame(rows)


def build_lift_report(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values("score_churn", ascending=False).reset_index(drop=True)
    total_rows = len(ranked)
    total_churners = int(ranked["y_real"].sum())
    baseline_rate = total_churners / total_rows

    rows = []
    for frac in TOP_K_FRACTIONS:
        n = max(1, int(round(total_rows * frac)))
        top = ranked.head(n)
        captured = int(top["y_real"].sum())
        precision_at_k = captured / n
        rows.append(
            {
                "top_fraction": frac,
                "top_percent": frac * 100,
                "contacted": n,
                "captured_churners": captured,
                "capture_rate": captured / total_churners if total_churners else np.nan,
                "precision_at_k": precision_at_k,
                "baseline_churn_rate": baseline_rate,
                "lift": precision_at_k / baseline_rate if baseline_rate else np.nan,
                "min_score": float(top["score_churn"].min()),
                "max_score": float(top["score_churn"].max()),
            }
        )
    return pd.DataFrame(rows)


def run_operational_evaluation() -> dict[str, Path]:
    ensure_dir(MODEL_REPORTS_DIR)
    df = load_scores()

    threshold_report = build_threshold_report(df)
    lift_report = build_lift_report(df)

    threshold_report.to_csv(THRESHOLD_REPORT, index=False)
    lift_report.to_csv(LIFT_REPORT, index=False)

    best_f1 = threshold_report.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0].to_dict()
    best_recall_50 = threshold_report[threshold_report["recall"] >= 0.50]
    best_recall_50_row = (
        best_recall_50.sort_values(["contacted", "precision"], ascending=[True, False]).iloc[0].to_dict()
        if not best_recall_50.empty
        else None
    )
    top_5 = lift_report[lift_report["top_fraction"] == 0.05].iloc[0].to_dict()
    top_10 = lift_report[lift_report["top_fraction"] == 0.10].iloc[0].to_dict()

    summary = {
        "iteration": "it_4",
        "source_scores": project_path(INPUT_SCORES),
        "rows": int(len(df)),
        "total_churners": int(df["y_real"].sum()),
        "baseline_churn_rate": float(df["y_real"].mean()),
        "best_threshold_by_f1": best_f1,
        "smallest_contact_volume_with_recall_at_least_50": best_recall_50_row,
        "top_5_percent": top_5,
        "top_10_percent": top_10,
        "notes": [
            "Iteracion 4 no reentrena modelos.",
            "Evalua umbrales y lift sobre los scores del mejor modelo de Iteracion 3C Top75.",
            "Usar estos resultados para decidir volumen operativo de contacto.",
        ],
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "threshold_report": THRESHOLD_REPORT,
        "lift_report": LIFT_REPORT,
        "summary": SUMMARY_JSON,
    }


if __name__ == "__main__":
    for name, path in run_operational_evaluation().items():
        print(f"{name}: {path}")
