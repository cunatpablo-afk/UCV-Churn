# Union Central Voz - Customer Churn Prediction

Proyecto de prediccion de churn mensual para clientes de una operadora de telecomunicaciones ficticia, con enfoque temporal y orientacion a decisiones de retencion.

## Objetivo

Construir un modelo que estime, para cada cliente y mes, la probabilidad de churn en el mes siguiente.

La unidad de analisis es:

```text
cliente_id + fecha mensual
```

El target de modelado es:

```text
churn_t_plus_1
```

Esto permite simular un uso realista: con informacion disponible hasta el cierre del mes `t`, se predice el riesgo de churn en `t+1`.

## Datos

El proyecto combina varias fuentes:

- Clientes.
- Churn mensual.
- Facturacion mensual.
- Calidad de red por zona y mes.
- Interacciones con soporte.
- Encuestas y senales de percepcion.

Los datos originales se conservan en `data/raw/` y las transformaciones reproducibles generan salidas en `data/processed/` y `data/modeling/`.

## Estructura

```text
config/      Configuracion del proyecto
data/        Datos raw, procesados y datasets de modelado
docs/        Decisiones, experimentos e informe final
models/      Modelos entrenados
notebooks/   EDA y analisis exploratorio
reports/     Metricas, figuras y resultados
src/         Codigo reproducible de limpieza, features y modelos
```

## Pipeline Principal

1. Limpieza de datos raw:

   ```bash
   python -m src.data.clean_data
   ```

2. Construccion del dataset cliente-mes:

   ```bash
   python -m src.features.build_modeling_dataset
   ```

3. Entrenamiento de modelos e iteraciones:

   ```bash
   python -m src.models.train_models_it_3g_logistic_semantic_imputation
   ```

## Modelo Recomendado

La version recomendada actualmente es:

| Elemento | Valor |
|---|---|
| Modelo | Logistic Regression |
| Iteracion | 3G - imputacion semantica |
| Dataset | `data/modeling/churn_modeling_dataset_it_3f_family_pruning.csv` |
| Uso recomendado | Ranking de riesgo y seleccion Top-K |
| Metrica principal | PR-AUC en test temporal |

Resultados principales en test temporal:

| Metrica | Valor |
|---|---:|
| PR-AUC | 0.0267 |
| ROC-AUC | 0.7132 |
| Recall | 0.5238 |
| Precision | 0.0136 |
| F1 | 0.0265 |

El modelo debe interpretarse como una herramienta de priorizacion, no como un clasificador binario cerrado. La politica operativa recomendada es usar segmentos Top-K, por ejemplo Top 5% o Top 10% de clientes con mayor score de churn.

## Documentacion

- Informe final: `docs/informe_modelado_churn.md`
- Registro de modelos: `docs/modelos.md`
- Registro de decisiones: `docs/decisiones.md`

## Notas Metodologicas

- El split principal es temporal, no aleatorio.
- La metrica principal es PR-AUC por el fuerte desbalance de clase.
- Se controla explicitamente el riesgo de leakage temporal.
- Las variables de identificacion directa y proxies geograficos mas claros se eliminan en las iteraciones finales.
- El test temporal se ha usado en varias iteraciones; antes de una puesta en produccion real conviene reservar un holdout temporal final intocable.
