from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier  # Desactivado en iteracion 2.
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import FIGURES_DIR, MODEL_REPORTS_DIR, MODELING_DIR, MODELS_DIR, ensure_dir, project_path

warnings.filterwarnings("ignore")

TARGET = "churn_t_plus_1"
SEED = 42
THRESHOLD = 0.5
RUN_SUFFIX = "it_2"

## Configuracion de seleccion del modelo.
## PR-AUC se usa como metrica principal porque churn es una clase minoritaria:
## mide mejor la capacidad de encontrar clientes con riesgo sin quedar cegados por
## la gran cantidad de no churn. Recall, F1 y ROC-AUC se usan como desempate y
## contexto, no como unica verdad.
PRIMARY_METRIC = "pr_auc_test"
SECONDARY_RANKING_METRICS = ["recall_test", "f1_test", "roc_auc_test"]

## Switch de busqueda de hiperparametros.
## - "none": entrena cada modelo con la configuracion base.
## - "randomized": prueba N_ITER combinaciones aleatorias por modelo.
## - "grid": prueba todas las combinaciones del grid; puede tardar bastante.
SEARCH_MODE = "randomized"  # opciones: "none", "randomized", "grid"
N_ITER = 5
CV_SPLITS = 3
N_JOBS_MODEL = 4
N_JOBS_SEARCH = 1
TRAIN_SAMPLE_FRAC = 1.0  # 1.0 usa todo el train; 0.20 acelera iteraciones de desarrollo.

## Los modelos se entrenan dentro de un Pipeline:
## Pipeline([("preprocessor", ...), ("model", estimador)])
## Por eso los parametros del estimador deben llevar el prefijo "model__".
## Ejemplo: RandomForestClassifier(max_depth=8) se escribe como
## "model__max_depth": [8].
PARAM_GRIDS = {
    # Iteracion 2: Logistic Regression queda fija con C=0.1.
    # "logistic_regression": {
    #     "model__C": [0.01, 0.1, 1, 10],
    # },
    # Iteracion 2: Random Forest se desactiva porque fue el cuello de botella.
    # "random_forest": {
    #     "model__n_estimators": [100, 200, 300],
    #     "model__max_depth": [5, 8, 12, None],
    #     "model__min_samples_leaf": [5, 10, 20, 30],
    #     "model__max_features": ["sqrt", "log2"],
    # },
    "xgboost": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9],
        "model__colsample_bytree": [0.7, 0.8, 0.9],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__reg_lambda": [1, 3, 5, 10],
        "model__reg_alpha": [0, 0.5, 1],
    },
}


def log(message: str) -> None:
    print(f"[train_models] {message}", flush=True)


def with_suffix(name: str) -> str:
    stem, extension = name.rsplit(".", 1)
    return f"{stem}_{RUN_SUFFIX}.{extension}"


def load_modeling_dataset() -> pd.DataFrame:
    path = MODELING_DIR / "churn_modeling_dataset.csv"
    log(f"Cargando dataset de modelado: {path}")
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero el feature engineering.")
    df = pd.read_csv(path, parse_dates=["fecha"])
    if TARGET not in df.columns:
        raise ValueError(f"No existe la variable objetivo {TARGET}")
    log(f"Dataset cargado: {len(df):,} filas, {df.shape[1]:,} columnas")
    return df


def temporal_split(df: pd.DataFrame, test_months: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    log(f"Preparando split temporal con {test_months} meses de test")
    ## Evaluacion principal temporal:
    ## entrenamos con meses antiguos y reservamos los ultimos meses como test.
    ## Esto simula mejor el caso real: usar informacion historica para predecir
    ## churn futuro, evitando mezclar meses futuros en entrenamiento.
    if "fecha" not in df.columns:
        raise ValueError("No existe la columna fecha. Es obligatoria para el split temporal.")
    if df["fecha"].isna().any():
        raise ValueError("La columna fecha contiene nulos. Revisar dataset de modelado.")
    months = sorted(df["fecha"].dropna().unique())
    if len(months) <= test_months:
        raise ValueError("No hay suficientes meses para un test temporal.")
    cutoff = months[-test_months]
    train = df[df["fecha"] < cutoff].copy()
    test = df[df["fecha"] >= cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Split temporal vacio.")
    log(
        "Split temporal listo: "
        f"train={len(train):,} filas hasta {train['fecha'].max().date()}, "
        f"test={len(test):,} filas desde {test['fecha'].min().date()} hasta {test['fecha'].max().date()}"
    )
    return train, test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    ## add_indicator=True crea columnas extra que marcan si un valor numerico era nulo.
    ## En este proyecto los nulos pueden ser informativos: por ejemplo, clientes nuevos
    ## sin historial suficiente para lags o rolling windows.
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def get_models(y_train: pd.Series) -> dict[str, Pipeline]:
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    base_models = {
        "dummy_mayoritaria": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(
            C=0.1,
            max_iter=5000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=N_JOBS_MODEL,
        ),
        # Iteracion 2: Random Forest queda desactivado por coste computacional.
        # "random_forest": RandomForestClassifier(
        #     n_estimators=150,
        #     max_depth=10,
        #     min_samples_leaf=25,
        #     class_weight="balanced",
        #     random_state=SEED,
        #     n_jobs=N_JOBS_MODEL,
        # ),
        # Iteracion 2: Gradient Boosting queda fuera del benchmark.
        # "gradient_boosting": GradientBoostingClassifier(
        #     n_estimators=120,
        #     learning_rate=0.05,
        #     max_depth=3,
        #     random_state=SEED,
        # ),
        "xgboost": XGBClassifier(
            n_estimators=160,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=N_JOBS_MODEL,
        ),
    }
    return base_models


def metric_dict(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float | str]:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "roc_auc": roc_auc_score(y_true, y_proba) if y_true.nunique() == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, y_proba) if y_true.nunique() == 2 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": json.dumps(cm.tolist()),
    }


def get_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [TARGET, "cliente_id", "fecha"]
    leakage_like = [c for c in df.columns if c.lower() in {"churn", "ever_churn"}]
    if leakage_like:
        raise ValueError(f"Columnas prohibidas detectadas: {leakage_like}")
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df[TARGET].astype(int).copy()
    return X, y


def sample_train_for_iteration(train_df: pd.DataFrame) -> pd.DataFrame:
    ## Muestreo opcional para desarrollar rapido.
    ## Con TRAIN_SAMPLE_FRAC=0.20 entrenamos con ~20% del train, manteniendo la
    ## proporcion de churn mediante muestreo estratificado. El test temporal NO se
    ## muestrea: siempre evaluamos contra el holdout temporal completo.
    if TRAIN_SAMPLE_FRAC >= 1.0:
        log("Usando train completo para entrenamiento")
        return train_df
    if TRAIN_SAMPLE_FRAC <= 0:
        raise ValueError("TRAIN_SAMPLE_FRAC debe ser mayor que 0")

    sampled = (
        train_df.groupby(TARGET, group_keys=False)
        .sample(frac=TRAIN_SAMPLE_FRAC, random_state=SEED)
        .sort_values(["fecha", "cliente_id"])
        .reset_index(drop=True)
    )
    log(
        "Usando muestra estratificada de train: "
        f"{len(sampled):,}/{len(train_df):,} filas ({TRAIN_SAMPLE_FRAC:.0%})"
    )
    return sampled


def tune_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    scoring_metric: str = "average_precision",
) -> tuple[Pipeline, dict[str, float | str]]:
    ## Esta funcion centraliza el tuning.
    ## Con SEARCH_MODE="none" no hace busqueda y solo entrena la configuracion base.
    ## Con "randomized" o "grid" busca hiperparametros usando average_precision,
    ## equivalente a PR-AUC en scikit-learn.
    if SEARCH_MODE not in {"none", "randomized", "grid"}:
        raise ValueError("SEARCH_MODE debe ser: 'none', 'randomized' o 'grid'")

    if SEARCH_MODE == "none" or model_name not in PARAM_GRIDS:
        log(f"{model_name}: tuning desactivado; entrenando configuracion base")
        model.fit(X_train, y_train)
        return model, {
            "model": model_name,
            "search_mode": SEARCH_MODE,
            "best_score": np.nan,
            "best_params": "{}",
        }

    params = PARAM_GRIDS[model_name]
    log(f"{model_name}: iniciando {SEARCH_MODE} search con metrica {scoring_metric}")

    if SEARCH_MODE == "randomized":
        ## RandomizedSearchCV es util para iterar rapido:
        ## no prueba todo el grid, solo N_ITER combinaciones aleatorias.
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=N_ITER,
            scoring=scoring_metric,
            cv=cv,
            random_state=SEED,
            n_jobs=N_JOBS_SEARCH,
            verbose=1,
            refit=True,
        )
    else:
        ## GridSearchCV prueba todas las combinaciones posibles.
        ## Es mas exhaustivo, pero puede ser muy lento con XGBoost/RandomForest.
        search = GridSearchCV(
            estimator=model,
            param_grid=params,
            scoring=scoring_metric,
            cv=cv,
            n_jobs=N_JOBS_SEARCH,
            verbose=1,
            refit=True,
        )

    search.fit(X_train, y_train)
    log(f"{model_name}: mejores parametros {search.best_params_}")
    log(f"{model_name}: mejor score CV ({scoring_metric})={search.best_score_:.4f}")

    return search.best_estimator_, {
        "model": model_name,
        "search_mode": SEARCH_MODE,
        "best_score": float(search.best_score_),
        "best_params": json.dumps(search.best_params_),
    }


def plot_curves(curve_data: dict[str, dict[str, np.ndarray]]) -> None:
    ensure_dir(FIGURES_DIR)

    plt.figure(figsize=(8, 6))
    for model_name, data in curve_data.items():
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_proba"])
        auc_value = roc_auc_score(data["y_true"], data["y_proba"])
        plt.plot(fpr, tpr, label=f"{model_name} ({auc_value:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.title("Curvas ROC por modelo")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / with_suffix("roc_curves.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    for model_name, data in curve_data.items():
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_proba"])
        ap = average_precision_score(data["y_true"], data["y_proba"])
        plt.plot(recall, precision, label=f"{model_name} ({ap:.3f})")
    plt.title("Curvas Precision-Recall por modelo")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / with_suffix("pr_curves.png"), dpi=160)
    plt.close()


def run_training() -> dict[str, Path]:
    log("Inicio del benchmark de modelos")
    ensure_dir(MODEL_REPORTS_DIR)
    ensure_dir(MODELS_DIR)

    df = load_modeling_dataset()
    train_df, test_df = temporal_split(df)
    full_train_rows = len(train_df)
    train_df = sample_train_for_iteration(train_df)
    X_train, y_train = get_feature_matrix(train_df)
    X_test, y_test = get_feature_matrix(test_df)
    log(f"Features entrenamiento: {X_train.shape[1]:,}")
    log(
        "Distribucion target train: "
        f"negativos={(y_train == 0).sum():,}, positivos={(y_train == 1).sum():,}, "
        f"tasa={y_train.mean():.4%}"
    )
    log(
        "Distribucion target test: "
        f"negativos={(y_test == 0).sum():,}, positivos={(y_test == 1).sum():,}, "
        f"tasa={y_test.mean():.4%}"
    )

    base_models = get_models(y_train)
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    log(f"Validacion interna: StratifiedKFold con {cv.get_n_splits()} folds")
    log(f"Fraccion de train usada: {TRAIN_SAMPLE_FRAC:.0%}")
    log(f"Metrica principal de ranking: {PRIMARY_METRIC}")
    log(f"Metricas secundarias de desempate: {', '.join(SECONDARY_RANKING_METRICS)}")
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
    }

    metrics_rows = []
    ranking_rows = []
    tuning_rows = []
    fitted_models: dict[str, Pipeline] = {}
    curve_data: dict[str, dict[str, np.ndarray]] = {}

    for name, estimator in base_models.items():
        log(f"Entrenando modelo: {name}")
        preprocessor = build_preprocessor(X_train)
        model = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

        ## Primero se selecciona la mejor configuracion segun SEARCH_MODE.
        ## Despues se evalua esa configuracion con las mismas metricas para todos
        ## los modelos, manteniendo comparable el ranking final.
        model, tuning_info = tune_model(
            model_name=name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            cv=cv,
            scoring_metric="average_precision",
        )
        tuning_rows.append(tuning_info)

        ## Esta CV es la evaluacion interna sobre train.
        ## El test temporal sigue siendo la evaluacion principal del proyecto.
        log(f"{name}: iniciando validacion cruzada final con la configuracion seleccionada")
        cv_result = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=N_JOBS_SEARCH)
        log(
            f"{name}: CV completada "
            f"PR-AUC={np.mean(cv_result['test_pr_auc']):.4f}, "
            f"ROC-AUC={np.mean(cv_result['test_roc_auc']):.4f}, "
            f"Recall={np.mean(cv_result['test_recall']):.4f}"
        )
        log(f"{name}: reentrenando modelo final en train completo")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]
        train_pred = (train_proba >= THRESHOLD).astype(int)
        test_pred = (test_proba >= THRESHOLD).astype(int)

        train_metrics = metric_dict(y_train, train_pred, train_proba)
        test_metrics = metric_dict(y_test, test_pred, test_proba)
        log(
            f"{name}: test temporal "
            f"PR-AUC={test_metrics['pr_auc']:.4f}, "
            f"ROC-AUC={test_metrics['roc_auc']:.4f}, "
            f"Precision={test_metrics['precision']:.4f}, "
            f"Recall={test_metrics['recall']:.4f}, "
            f"F1={test_metrics['f1']:.4f}"
        )
        metrics_rows.append({"model": name, "split": "train", **train_metrics})
        metrics_rows.append(
            {
                "model": name,
                "split": "cv_mean",
                "roc_auc": float(np.mean(cv_result["test_roc_auc"])),
                "pr_auc": float(np.mean(cv_result["test_pr_auc"])),
                "accuracy": np.nan,
                "precision": float(np.mean(cv_result["test_precision"])),
                "recall": float(np.mean(cv_result["test_recall"])),
                "f1": float(np.mean(cv_result["test_f1"])),
                "confusion_matrix": "",
            }
        )
        metrics_rows.append({"model": name, "split": "test_temporal", **test_metrics})
        ranking_rows.append(
            {
                "model": name,
                "pr_auc_test": test_metrics["pr_auc"],
                "recall_test": test_metrics["recall"],
                "f1_test": test_metrics["f1"],
                "roc_auc_test": test_metrics["roc_auc"],
            }
        )
        curve_data[name] = {"y_true": y_test.to_numpy(), "y_proba": test_proba}

    metrics = pd.DataFrame(metrics_rows)

    ## Ranking final:
    ## se ordena primero por PR-AUC en el test temporal, y luego por recall, F1
    ## y ROC-AUC como criterios secundarios. Asi PR-AUC pesa mas, pero no decide
    ## completamente aislada del resto de metricas.
    ranking = (
        pd.DataFrame(ranking_rows)
        .sort_values([PRIMARY_METRIC, *SECONDARY_RANKING_METRICS], ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)

    best_name = ranking.loc[0, "model"]
    log(f"Modelo ganador por ranking: {best_name}")
    best_model = fitted_models[best_name]
    best_model_path = MODELS_DIR / with_suffix("best_model.joblib")
    log(f"Guardando modelo ganador en {best_model_path}")
    joblib.dump(best_model, best_model_path)
    log("Generando curvas ROC y Precision-Recall")
    plot_curves(curve_data)

    metrics_path = MODEL_REPORTS_DIR / with_suffix("model_metrics.csv")
    ranking_path = MODEL_REPORTS_DIR / with_suffix("model_ranking.csv")
    tuning_path = MODEL_REPORTS_DIR / with_suffix("model_tuning_results.csv")
    test_scores_path = MODEL_REPORTS_DIR / with_suffix("test_scores_best_model.csv")
    metrics.to_csv(metrics_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    pd.DataFrame(tuning_rows).to_csv(tuning_path, index=False)

    ## Este CSV permite analizar cliente a cliente el resultado del mejor modelo:
    ## - y_real: churn real en t+1.
    ## - score_churn: probabilidad estimada de churn.
    ## - pred_churn: decision binaria usando THRESHOLD.
    ## Es la base para analisis de falsos positivos/falsos negativos y priorizacion.
    best_test_proba = curve_data[best_name]["y_proba"]
    test_scores = test_df[["cliente_id", "fecha", TARGET]].copy()
    test_scores = test_scores.rename(columns={TARGET: "y_real"})
    test_scores["score_churn"] = best_test_proba
    test_scores["pred_churn"] = (test_scores["score_churn"] >= THRESHOLD).astype(int)
    test_scores.to_csv(test_scores_path, index=False)
    log(f"Scores del mejor modelo guardados en {test_scores_path}")

    summary = {
        "best_model": best_name,
        "run_suffix": RUN_SUFFIX,
        "target": TARGET,
        "threshold": THRESHOLD,
        "search_mode": SEARCH_MODE,
        "n_iter": N_ITER if SEARCH_MODE == "randomized" else None,
        "cv_splits": CV_SPLITS,
        "n_jobs_model": N_JOBS_MODEL,
        "n_jobs_search": N_JOBS_SEARCH,
        "primary_metric": PRIMARY_METRIC,
        "secondary_ranking_metrics": SECONDARY_RANKING_METRICS,
        "secondary_report_metrics": ["roc_auc", "accuracy", "precision", "recall", "f1", "confusion_matrix"],
        "train_rows_full": int(full_train_rows),
        "train_rows_used": int(len(train_df)),
        "train_sample_frac": TRAIN_SAMPLE_FRAC,
        "test_rows": int(len(test_df)),
        "test_month_min": str(test_df["fecha"].min().date()),
        "test_month_max": str(test_df["fecha"].max().date()),
        "test_scores": project_path(test_scores_path),
    }
    summary_path = MODEL_REPORTS_DIR / with_suffix("training_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Metricas guardadas en {metrics_path}")
    log(f"Ranking guardado en {ranking_path}")
    log(f"Resultados de tuning guardados en {tuning_path}")
    log("Benchmark terminado correctamente")

    return {
        "metrics": metrics_path,
        "ranking": ranking_path,
        "tuning": tuning_path,
        "best_model": best_model_path,
        "test_scores": test_scores_path,
        "roc_curve": FIGURES_DIR / with_suffix("roc_curves.png"),
        "pr_curve": FIGURES_DIR / with_suffix("pr_curves.png"),
        "summary": summary_path,
    }


if __name__ == "__main__":
    for name, path in run_training().items():
        print(f"{name}: {path}")
