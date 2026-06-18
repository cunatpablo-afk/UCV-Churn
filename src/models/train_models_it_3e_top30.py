from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.train_models as base
from src.models.export_feature_importance_it_3d_top50 import extract_original_feature, get_model_importance
from src.utils.paths import FIGURES_DIR, MODEL_REPORTS_DIR, MODELING_DIR, MODELS_DIR, ensure_dir

## Iteracion 3E:
## Dataset Top30 desde la importancia real de 3D Top50.
## Mantiene cliente_id, fecha y target en el CSV, pero base.get_feature_matrix
## los excluye del entrenamiento.
## Sin tuning: mismos hiperparametros que 3D para aislar el efecto de reducir
## variables y analizar multicolinealidad.

RUN_SUFFIX = "it_3e_top30"
DATASET = MODELING_DIR / "churn_modeling_dataset_it_3e_top30.csv"

base.RUN_SUFFIX = RUN_SUFFIX
base.SEARCH_MODE = "none"
base.N_ITER = 5
base.TRAIN_SAMPLE_FRAC = 1.0


def load_modeling_dataset() -> pd.DataFrame:
    base.log(f"Cargando dataset Top30 de modelado: {DATASET}")
    if not DATASET.exists():
        raise FileNotFoundError(
            f"No existe {DATASET}. Ejecuta primero src/features/build_reduced_dataset_it_3e_top30.py."
        )
    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    if base.TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo {base.TARGET}")
    base.log(f"Dataset Top30 cargado: {len(df):,} filas, {df.shape[1]:,} columnas")
    return df


def get_models(y_train: pd.Series):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    return {
        "dummy_mayoritaria": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            C=0.1,
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=base.SEED,
            n_jobs=base.N_JOBS_MODEL,
        ),
        "xgboost": XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.9,
            min_child_weight=5,
            reg_lambda=3,
            reg_alpha=1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=base.SEED,
            n_jobs=base.N_JOBS_MODEL,
        ),
    }


def _diagnose_gap(train_value: float, cv_value: float, test_value: float) -> str:
    if pd.isna(train_value) or pd.isna(cv_value) or pd.isna(test_value):
        return "ok"
    if train_value - cv_value > 0.03 and train_value - test_value > 0.03:
        return "overfitting"
    if max(train_value, cv_value, test_value) < 0.02:
        return "underfitting"
    return "ok"


def export_overfitting_report() -> Path:
    metrics_path = MODEL_REPORTS_DIR / f"model_metrics_{RUN_SUFFIX}.csv"
    output_path = MODEL_REPORTS_DIR / f"overfitting_diagnostics_{RUN_SUFFIX}.csv"
    metrics = pd.read_csv(metrics_path)
    rows = []
    for model_name, group in metrics.groupby("model"):
        by_split = group.set_index("split")
        train = by_split.loc["train"]
        cv = by_split.loc["cv_mean"]
        test = by_split.loc["test_temporal"]
        row = {
            "model": model_name,
            "gap_train_cv_roc_auc": float(train["roc_auc"] - cv["roc_auc"]),
            "gap_train_test_roc_auc": float(train["roc_auc"] - test["roc_auc"]),
            "gap_cv_test_roc_auc": float(cv["roc_auc"] - test["roc_auc"]),
            "gap_train_cv_pr_auc": float(train["pr_auc"] - cv["pr_auc"]),
            "gap_train_test_pr_auc": float(train["pr_auc"] - test["pr_auc"]),
            "gap_cv_test_pr_auc": float(cv["pr_auc"] - test["pr_auc"]),
        }
        row["diagnostico"] = _diagnose_gap(train["pr_auc"], cv["pr_auc"], test["pr_auc"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def export_feature_importance() -> dict[str, Path]:
    ensure_dir(MODEL_REPORTS_DIR)
    model_path = MODELS_DIR / f"best_model_{RUN_SUFFIX}.joblib"
    model = joblib.load(model_path)
    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    train_df, _ = base.temporal_split(df)
    X_train, _ = base.get_feature_matrix(train_df)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    importance_values, importance_type = get_model_importance(model)
    if len(feature_names) != len(importance_values):
        raise ValueError(f"No coinciden features e importancias: {len(feature_names)} vs {len(importance_values)}")

    importance = pd.DataFrame(
        {
            "feature_transformed": feature_names,
            "importance": importance_values,
            "abs_importance": np.abs(importance_values),
            "importance_type": importance_type,
        }
    )
    importance["feature_original"] = importance["feature_transformed"].map(
        lambda value: extract_original_feature(value, X_train.columns.tolist())
    )
    importance = importance.sort_values("abs_importance", ascending=False).reset_index(drop=True)
    importance["rank"] = np.arange(1, len(importance) + 1)

    grouped = (
        importance.groupby("feature_original", as_index=False)
        .agg(
            abs_importance=("abs_importance", "sum"),
            max_abs_importance=("abs_importance", "max"),
            n_transformed_features=("feature_transformed", "count"),
        )
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["rank"] = np.arange(1, len(grouped) + 1)

    importance_path = MODEL_REPORTS_DIR / f"feature_importance_{RUN_SUFFIX}.csv"
    grouped_path = MODEL_REPORTS_DIR / f"feature_importance_grouped_{RUN_SUFFIX}.csv"
    importance.to_csv(importance_path, index=False)
    grouped.to_csv(grouped_path, index=False)
    return {"feature_importance": importance_path, "feature_importance_grouped": grouped_path}


def _calculate_vif(numeric_df: pd.DataFrame) -> pd.DataFrame:
    clean = numeric_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    clean = clean.fillna(clean.median(numeric_only=True))
    clean = clean.loc[:, clean.nunique(dropna=False) > 1]

    rows = []
    for col in clean.columns:
        y = clean[col].to_numpy()
        X = clean.drop(columns=[col]).to_numpy()
        if X.shape[1] == 0:
            vif = 1.0
        else:
            r2 = LinearRegression().fit(X, y).score(X, y)
            vif = np.inf if r2 >= 0.999999 else 1.0 / max(1.0 - r2, 1e-12)
        rows.append({"feature": col, "vif": float(vif)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def export_multicollinearity_report() -> dict[str, Path]:
    ensure_dir(MODEL_REPORTS_DIR)
    ensure_dir(FIGURES_DIR)
    df = pd.read_csv(DATASET, parse_dates=["fecha"])
    X, _ = base.get_feature_matrix(df)
    numeric = X.select_dtypes(include=[np.number, "bool"]).copy()

    corr = numeric.corr()
    corr_path = MODEL_REPORTS_DIR / f"feature_correlation_matrix_{RUN_SUFFIX}.csv"
    corr.to_csv(corr_path)

    pairs = []
    cols = corr.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and abs(value) > 0.80:
                pairs.append(
                    {
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": float(value),
                        "abs_correlation": float(abs(value)),
                        "above_0_90": bool(abs(value) > 0.90),
                    }
                )
    pairs_path = MODEL_REPORTS_DIR / f"high_correlation_pairs_{RUN_SUFFIX}.csv"
    pairs_df = pd.DataFrame(
        pairs,
        columns=["feature_a", "feature_b", "correlation", "abs_correlation", "above_0_90"],
    )
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("abs_correlation", ascending=False)
    pairs_df.to_csv(pairs_path, index=False)

    vif_path = MODEL_REPORTS_DIR / f"feature_vif_{RUN_SUFFIX}.csv"
    _calculate_vif(numeric).to_csv(vif_path, index=False)

    heatmap_path = FIGURES_DIR / f"correlation_heatmap_{RUN_SUFFIX}.png"
    if not corr.empty:
        fig_size = max(8, min(18, len(corr.columns) * 0.45))
        plt.figure(figsize=(fig_size, fig_size))
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(label="Correlacion")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
        plt.title("Correlacion variables numericas Top30")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=160)
        plt.close()

    return {
        "correlation_matrix": corr_path,
        "high_correlation_pairs": pairs_path,
        "vif": vif_path,
        "correlation_heatmap": heatmap_path,
    }


def export_iteration_comparison() -> Path:
    rows = []
    configs = [
        ("3C Top75", "it_3c_top75", 75),
        ("3D Top50", "it_3d_top50", 50),
        ("3E Top30", RUN_SUFFIX, 30),
    ]
    diagnostics_path = MODEL_REPORTS_DIR / f"overfitting_diagnostics_{RUN_SUFFIX}.csv"
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    for label, suffix, n_features in configs:
        ranking_path = MODEL_REPORTS_DIR / f"model_ranking_{suffix}.csv"
        metrics_path = MODEL_REPORTS_DIR / f"model_metrics_{suffix}.csv"
        if not ranking_path.exists():
            continue
        ranking = pd.read_csv(ranking_path).sort_values("rank")
        best = ranking.iloc[0]
        model_name = best["model"]
        precision = np.nan
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            match_metrics = metrics[(metrics["model"] == model_name) & (metrics["split"] == "test_temporal")]
            if not match_metrics.empty:
                precision = match_metrics.iloc[0]["precision"]
        diagnostic = ""
        if suffix == RUN_SUFFIX and not diagnostics.empty:
            match = diagnostics[diagnostics["model"] == model_name]
            if not match.empty:
                diagnostic = str(match.iloc[0]["diagnostico"])
        rows.append(
            {
                "iteration": label,
                "suffix": suffix,
                "best_model": model_name,
                "n_features": n_features,
                "pr_auc_test": best["pr_auc_test"],
                "roc_auc_test": best["roc_auc_test"],
                "recall_test": best["recall_test"],
                "precision_test": precision,
                "f1_test": best["f1_test"],
                "feature_reduction_vs_3c": 75 - n_features,
                "overfitting_diagnostic": diagnostic,
            }
        )
    output_path = MODEL_REPORTS_DIR / f"iteration_comparison_{RUN_SUFFIX}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


base.load_modeling_dataset = load_modeling_dataset
base.get_models = get_models


if __name__ == "__main__":
    outputs = base.run_training()
    outputs["overfitting"] = export_overfitting_report()
    outputs.update(export_feature_importance())
    outputs.update(export_multicollinearity_report())
    outputs["comparison"] = export_iteration_comparison()
    for name, path in outputs.items():
        print(f"{name}: {path}")
