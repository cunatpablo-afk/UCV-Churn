from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import MODEL_REPORTS_DIR, MODELING_DIR, ensure_dir, project_path

INPUT_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3d_top50.csv"
TOP_FEATURES = MODEL_REPORTS_DIR / "top_features_30_it_3d_top50.csv"
OUTPUT_DATASET = MODELING_DIR / "churn_modeling_dataset_it_3e_top30.csv"
REPORT_CSV = MODELING_DIR / "feature_selection_report_it_3e_top30.csv"
REPORT_JSON = MODELING_DIR / "feature_selection_summary_it_3e_top30.json"

ID_AND_TARGET_COLS = ["cliente_id", "fecha", "churn_t_plus_1"]
TOP_N = 30
GEO_PROXY_TERMS = ["poblacion_zona"]


def build_reduced_dataset() -> dict[str, Path]:
    ensure_dir(MODELING_DIR)
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"No existe {INPUT_DATASET}. Ejecuta primero 3D Top50.")
    if not TOP_FEATURES.exists():
        raise FileNotFoundError(
            f"No existe {TOP_FEATURES}. Ejecuta primero src/models/export_feature_importance_it_3d_top50.py."
        )

    df = pd.read_csv(INPUT_DATASET, parse_dates=["fecha"])
    top = pd.read_csv(TOP_FEATURES).head(TOP_N)
    top_features = [col for col in top["feature_original"].tolist() if col not in ID_AND_TARGET_COLS]

    missing = [col for col in top_features if col not in df.columns]
    if missing:
        raise ValueError(f"Top features no encontradas en dataset 3D: {missing}")

    selected_columns = ID_AND_TARGET_COLS + top_features
    reduced = df[selected_columns].copy()
    reduced.to_csv(OUTPUT_DATASET, index=False)

    selected_set = set(selected_columns)
    rows = []
    for col in df.columns:
        rows.append(
            {
                "column": col,
                "selected": int(col in selected_set),
                "reason": "top30_it_3d" if col in top_features else "id_target" if col in ID_AND_TARGET_COLS else "fuera_top30",
            }
        )
    pd.DataFrame(rows).to_csv(REPORT_CSV, index=False)

    geo_proxy_features = [
        col for col in top_features if any(term in col.lower() for term in GEO_PROXY_TERMS)
    ]
    summary = {
        "iteration": "it_3e_top30",
        "input_dataset": project_path(INPUT_DATASET),
        "output_dataset": project_path(OUTPUT_DATASET),
        "source_importance": project_path(TOP_FEATURES),
        "selection_rule": "top 30 features originales agrupadas desde feature importance de Iteracion 3D Top50",
        "rows": int(len(reduced)),
        "input_columns_total": int(len(df.columns)),
        "selected_columns_total": int(len(selected_columns)),
        "selected_feature_columns": int(len(top_features)),
        "removed_feature_columns": int(len([c for c in df.columns if c not in selected_set and c not in ID_AND_TARGET_COLS])),
        "id_target_columns_kept": ID_AND_TARGET_COLS,
        "model_feature_columns": top_features,
        "geo_proxy_features_detected": geo_proxy_features,
        "important_notes_for_results_review": [
            "cliente_id, fecha y churn_t_plus_1 se conservan en el CSV, pero no entran como features.",
            "Comparacion justa contra 3C Top75 y 3D Top50: mismos modelos, split temporal y sin tuning.",
            "Correlacion y VIF se calculan solo sobre variables numericas Top30.",
            "Revisar proxies geograficos antes de adopcion final.",
        ],
        "selected_features": top_features,
        "removed_columns": [c for c in df.columns if c not in selected_set],
    }
    REPORT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "dataset": OUTPUT_DATASET,
        "report_csv": REPORT_CSV,
        "report_json": REPORT_JSON,
    }


if __name__ == "__main__":
    for name, path in build_reduced_dataset().items():
        print(f"{name}: {path}")
