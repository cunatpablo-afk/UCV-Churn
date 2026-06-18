# Registro De Modelos

Este documento resume los experimentos de modelado. Las metricas detalladas se generan en:

- `reports/models/model_metrics.csv`
- `reports/models/model_ranking.csv`
- `reports/models/training_summary.json`

## Experimento 001 - Benchmark End-To-End

- Fecha: 2026-06-16
- Dataset: `data/modeling/churn_modeling_dataset.csv`
- Target: `churn_t_plus_1`
- Split: test temporal con los ultimos meses disponibles.
- Modelos:
  - Dummy baseline.
  - Logistic Regression.
  - Random Forest.
  - Gradient Boosting.
  - XGBoost.
- Criterio de seleccion:
  1. PR-AUC en test temporal.
  2. Recall de churn.
  3. F1 de churn.

## Resultado

Entrenamiento ejecutado el 2026-06-16.

- Filas train: 263099.
- Filas test temporal: 48888.
- Ventana test temporal: 2025-06-01 a 2025-11-01.
- Modelo ganador segun ranking automatico: `gradient_boosting`.
- Criterio aplicado: mayor PR-AUC en test temporal, seguido de recall y F1.

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| Gradient Boosting | 0.0275 | 0.0040 | 0.0079 | 0.7231 |
| Logistic Regression | 0.0226 | 0.5159 | 0.0252 | 0.6958 |
| Random Forest | 0.0175 | 0.1190 | 0.0457 | 0.6574 |
| XGBoost | 0.0172 | 0.3651 | 0.0325 | 0.6690 |
| Dummy | 0.0052 | 0.0000 | 0.0000 | 0.5000 |

## Lectura Critica

El `gradient_boosting` queda primero por PR-AUC, pero con umbral 0.5 casi no detecta churners en test temporal: recall 0.004. Esto lo hace poco util como sistema operativo de retencion sin ajuste de threshold.

Para una accion comercial donde interesa detectar clientes en riesgo, `logistic_regression` es el candidato operativo inicial mas razonable: menor PR-AUC que Gradient Boosting, pero recall mucho mayor. XGBoost tambien detecta mas churners que Gradient Boosting, aunque con peor precision/F1.

Decision para siguiente iteracion: evaluar ajuste de threshold y comparar modelos con una metrica orientada a negocio, por ejemplo recall minimo aceptable y precision esperada.

## Limitaciones A Revisar

- Posible mejora de variables temporales por fuente.
- Analisis de falsos negativos por segmentos.
- Ajuste de threshold segun coste de retencion.
- Comparacion incremental por familias de variables.
- Clase positiva muy minoritaria en el dataset mensual: aproximadamente 0.64%.
- El umbral 0.5 no es adecuado para todos los modelos.

## Experimento 002 - Iteracion 1 Ligera Sin Tuning

- Fecha: 2026-06-16.
- Dataset: `data/modeling/churn_modeling_dataset.csv`.
- Target: `churn_t_plus_1`.
- Objetivo: validar el pipeline completo con una corrida rapida y estable.
- Split: test temporal completo desde 2025-06-01 hasta 2025-11-01.
- Entrenamiento: muestra estratificada del 20% del train.
- Filas train completas: 263099.
- Filas train usadas: 52620.
- Filas test temporal: 48888.
- Search mode: `none`.
- CV interna: `StratifiedKFold` con 3 folds.
- Modelos:
  - Dummy baseline.
  - Logistic Regression.
  - Random Forest.
  - XGBoost.

### Resultado Iteracion 1

Modelo ganador automatico: `logistic_regression`, por mayor PR-AUC en test temporal.

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0125 | 0.4167 | 0.0177 | 0.6505 |
| XGBoost | 0.0121 | 0.2341 | 0.0287 | 0.6326 |
| Random Forest | 0.0120 | 0.0000 | 0.0000 | 0.6295 |
| Dummy | 0.0052 | 0.0000 | 0.0000 | 0.5000 |

### Lectura Critica Iteracion 1

El baseline dummy tiene PR-AUC 0.0052, practicamente igual a la tasa de churn del test temporal. Esto confirma que la metrica esta bien calibrada para un problema muy desbalanceado.

La regresion logistica queda primera por PR-AUC y detecta 105 de 252 churners en test temporal, con recall 0.4167. El coste es una precision muy baja: 0.0090. Es decir, recupera bastantes churners, pero marcando muchos falsos positivos.

XGBoost queda muy cerca en PR-AUC y obtiene mejor F1 que la regresion logistica, porque predice menos positivos y mejora la precision relativa. Detecta 59 de 252 churners, con recall 0.2341 y precision 0.0153. Puede ser mejor candidato si el coste de contactar falsos positivos importa mas.

Random Forest generaliza mal en esta configuracion: en train tiene metricas muy altas, pero en test temporal no detecta ningun churner con umbral 0.5. Esto sugiere sobreajuste y/o probabilidades demasiado conservadoras. No conviene descartarlo definitivamente, pero no es buen candidato operativo sin ajuste de threshold o configuracion mas restrictiva.

### Decision

Esta iteracion sirve como prueba rapida de pipeline, no como seleccion final. Para la siguiente iteracion conviene:

- Mantener test temporal completo.
- Probar `TRAIN_SAMPLE_FRAC = 1.0` con `SEARCH_MODE = "none"` si el tiempo lo permite.
- Analizar thresholds alternativos para Logistic Regression y XGBoost.
- Priorizar PR-AUC, Recall y F1, no Accuracy.
- Revisar falsos positivos y falsos negativos por segmento antes de decidir modelo operativo.

## Experimento 003 - Iteracion 2 Con Train Completo Y XGBoost Tuning

- Fecha: 2026-06-17.
- Dataset: `data/modeling/churn_modeling_dataset.csv`.
- Target: `churn_t_plus_1`.
- Objetivo: entrenar con el 100% del train, simplificando el benchmark y concentrando el tuning en XGBoost.
- Split: test temporal completo desde 2025-06-01 hasta 2025-11-01.
- Filas train usadas: 263099.
- Filas test temporal: 48888.
- Search mode: `randomized`.
- Iteraciones randomized: 5.
- CV interna: `StratifiedKFold` con 3 folds.
- Modelos:
  - Dummy baseline.
  - Logistic Regression fija con `C=0.1`.
  - XGBoost con randomized search.
- Modelos desactivados:
  - Random Forest, por coste computacional y bajo rendimiento operativo anterior.
  - Gradient Boosting, para mantener una iteracion mas enfocada.

### Resultado Iteracion 2

Modelo ganador automatico: `logistic_regression`, por mayor PR-AUC en test temporal.

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0237 | 0.5119 | 0.0252 | 0.6983 |
| XGBoost | 0.0232 | 0.4206 | 0.0284 | 0.7079 |
| Dummy | 0.0052 | 0.0000 | 0.0000 | 0.5000 |

### Configuracion Ganadora De XGBoost

Mejor score CV de XGBoost: PR-AUC 0.0199.

```text
n_estimators=100
max_depth=3
learning_rate=0.03
subsample=0.7
colsample_bytree=0.9
min_child_weight=5
reg_lambda=3
reg_alpha=1
```

La configuracion seleccionada es conservadora y regularizada, lo cual es coherente con un problema muy desbalanceado y con muchas variables.

### Lectura Critica Iteracion 2

La regresion logistica queda primera por PR-AUC, pero la diferencia con XGBoost es pequena: 0.0237 frente a 0.0232. Logistic detecta mas churners en test temporal: 129 de 252, con recall 0.5119. El coste es una precision muy baja: 0.0129, con 9837 falsos positivos.

XGBoost obtiene mejor ROC-AUC y mejor F1. Detecta 106 de 252 churners, con recall 0.4206 y precision 0.0147. Aunque captura menos churners, genera menos falsos positivos: 7118. Puede ser mas atractivo si el coste de contactar clientes erroneamente marcados como riesgo es relevante.

El dummy confirma el baseline esperado: PR-AUC 0.0052, equivalente aproximadamente a la tasa de churn del test temporal.

### Overfitting

No se observa overfitting grave en Logistic ni en XGBoost.

| Modelo | PR-AUC train | PR-AUC CV | PR-AUC test |
|---|---:|---:|---:|
| Logistic Regression | 0.0224 | 0.0181 | 0.0237 |
| XGBoost | 0.0277 | 0.0199 | 0.0232 |

XGBoost muestra algo mas de diferencia entre train y CV, pero el test temporal no cae respecto a CV. Esto sugiere que la regularizacion elegida esta funcionando razonablemente.

### Feature Importance

Se generaron los archivos:

- `reports/models/feature_importance_it_2.csv`
- `reports/models/feature_importance_grouped_it_2.csv`
- `reports/models/top_features_50_it_2.csv`

Top variables agrupadas del modelo ganador:

| Rank | Variable |
|---:|---|
| 1 | `fact_zona_id` |
| 2 | `zona_id` |
| 3 | `soporte_canal_principal` |
| 4 | `tipo_zona` |
| 5 | `red_tipo_zona` |
| 6 | `fact_stress_calidad_lag_lag_2m` |
| 7 | `region` |
| 8 | `red_region` |
| 9 | `fact_importe_total_lag_1m` |
| 10 | `fact_importe_total` |
| 11 | `fact_cargo_base_lag_2m` |
| 12 | `fact_stress_calidad_lag_roll_3m` |
| 13 | `red_indice_calidad_global_lag_2m` |
| 14 | `red_cobertura_5g_pct_lag_2m` |
| 15 | `soporte_motivo_principal` |

Las variables de zona pesan mucho. Puede ser senal real, pero tambien hay redundancia entre `zona_id`, `fact_zona_id`, `region` y `red_region`. Para simplificar no conviene tomar las top features de forma mecanica: es mejor seleccionar variables agrupadas y reducir duplicidades.

### Decision

Para la siguiente iteracion se propone:

- Probar thresholds alternativos para Logistic Regression y XGBoost.
- Crear una version `it_3` con top features agrupadas, por ejemplo top 20 o top 30.
- Evitar duplicidad excesiva de variables de zona.
- Comparar modelo completo frente a modelo reducido usando el mismo test temporal.
- Priorizar PR-AUC, Recall, F1 y matriz de confusion; no decidir por Accuracy.

## Experimento 004 - Iteracion 3A Sin Variables Geograficas Directas

- Fecha: 2026-06-17.
- Dataset reducido: `data/modeling/churn_modeling_dataset_it_3a.csv`.
- Dataset original base: `data/modeling/churn_modeling_dataset.csv`.
- Target: `churn_t_plus_1`.
- Objetivo: comprobar cuanto rendimiento se mantiene usando solo variables de facturacion, soporte y calidad de red, excluyendo identificadores geograficos directos.
- Regla de seleccion: prefijos `fact_`, `soporte_`, `red_`, con blacklist geografica explicita.
- Variables originales: 243 features de modelado.
- Variables seleccionadas: 183 features.
- Variables eliminadas: 60 features.
- Split: mismo test temporal, desde 2025-06-01 hasta 2025-11-01.
- Train usado: 263099 filas.
- Test temporal: 48888 filas.
- Search mode: `none`.
- Modelos:
  - Dummy baseline.
  - Logistic Regression fija con `C=0.1`.
  - XGBoost con hiperparametros ganadores de Iteracion 2.

Archivos generados:

- `data/modeling/feature_selection_report_it_3a.csv`
- `data/modeling/feature_selection_summary_it_3a.json`
- `reports/models/model_metrics_it_3a.csv`
- `reports/models/model_ranking_it_3a.csv`
- `reports/models/training_summary_it_3a.json`
- `reports/models/test_scores_best_model_it_3a.csv`
- `models/best_model_it_3a.joblib`

### Resultado Iteracion 3A

Modelo ganador automatico: `logistic_regression`, por mayor PR-AUC en test temporal.

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0238 | 0.5238 | 0.0246 | 0.6992 |
| XGBoost | 0.0219 | 0.4008 | 0.0308 | 0.6880 |
| Dummy | 0.0052 | 0.0000 | 0.0000 | 0.5000 |

### Comparacion Contra Iteracion 2

| Modelo | PR-AUC it_2 | PR-AUC it_3A | Cambio |
|---|---:|---:|---:|
| Logistic Regression | 0.0237 | 0.0238 | +0.0001 |
| XGBoost | 0.0232 | 0.0219 | -0.0014 |

La regresion logistica mantiene practicamente el mismo rendimiento que en el dataset completo, incluso tras eliminar variables geograficas directas y reducir el numero de features. Esto indica que gran parte de la senal util para Logistic proviene de facturacion, soporte y calidad de red.

XGBoost pierde algo de PR-AUC respecto a Iteracion 2, aunque mejora F1 frente a Logistic en esta iteracion. La caida es moderada y puede deberse a que los hiperparametros ganadores de Iteracion 2 estaban ajustados para el dataset completo.

### Criterio 90-95%

Referencia Logistic Iteracion 2: PR-AUC 0.0237.

- 90% de referencia: 0.0213.
- 95% de referencia: 0.0225.
- Logistic Iteracion 3A: 0.0238.

La reduccion es exitosa para Logistic segun el criterio definido: mantiene mas del 95% del PR-AUC de Iteracion 2 usando menos variables y eliminando geografia directa.

### Overfitting

| Modelo | PR-AUC train | PR-AUC CV | PR-AUC test |
|---|---:|---:|---:|
| Logistic Regression | 0.0200 | 0.0176 | 0.0238 |
| XGBoost | 0.0266 | 0.0180 | 0.0219 |

No se observa overfitting grave. XGBoost mantiene una brecha mayor entre train y CV que Logistic, pero el test temporal sigue por encima del CV. Logistic parece mas estable y menos dependiente de variables especificas.

### Lectura Critica Iteracion 3A

La eliminacion de variables geograficas directas no perjudica a Logistic; de hecho su PR-AUC sube marginalmente. Esto refuerza la idea de que el modelo puede ser simplificado sin perder rendimiento y con mejor explicabilidad.

Matiz importante: 3A elimina identificadores geograficos directos (`zona_id`, `region`, `tipo_zona`, etc.), pero no elimina todos los posibles proxies geograficos. En concreto, variables como `red_poblacion_zona` y sus lags entraron por el prefijo `red_`. No identifican la zona, pero si describen contexto de zona y deben tratarse como proxy geografico.

XGBoost baja respecto a Iteracion 2, pero conserva un F1 superior al de Logistic: 0.0308 frente a 0.0246. Esto ocurre porque XGBoost genera menos falsos positivos, aunque captura menos churners.

Confusiones principales en test temporal:

- Logistic Regression: 132 churners detectados, 120 falsos negativos, 10360 falsos positivos.
- XGBoost: 101 churners detectados, 151 falsos negativos, 6202 falsos positivos.

Si el objetivo es maximizar cobertura de churners, Logistic sigue siendo mas atractiva. Si el objetivo es reducir contactos innecesarios, XGBoost reducido puede ser mas razonable.

### Recordatorios Para La Siguiente Comparacion

- Iteracion 3A: sin variables geograficas directas.
- Iteracion 3B: anadir geografia agregada no identificativa, por ejemplo calidad/facturacion/soporte agregados por zona o region, pero sin IDs directos.
- La seleccion por nombre es practica pero imperfecta; revisar columnas conservadas y eliminadas antes de interpretar causalmente.
- Comparacion principal de 3B: mismos hiperparametros que Iteracion 2.
- Comparacion secundaria opcional: pequeno tuning solo de XGBoost reducido si el rendimiento cae o queda cerca.

### Decision

Adoptar Iteracion 3A como baseline simplificado inicial para interpretabilidad. La siguiente prueba recomendada es Iteracion 3B, incorporando solo contexto geografico agregado y no identificativo para comprobar si aporta senal adicional sin volver a depender de identificadores directos de zona.

## Experimento 005 - Iteracion 3B Con Geografia Agregada No Identificativa

- Fecha: 2026-06-17.
- Dataset reducido: `data/modeling/churn_modeling_dataset_it_3b.csv`.
- Dataset original base: `data/modeling/churn_modeling_dataset.csv`.
- Target: `churn_t_plus_1`.
- Objetivo: comprobar si anadir contexto geografico agregado mejora el dataset 3A sin reintroducir identificadores directos de zona o region.
- Base: variables de Iteracion 3A.
- Variables geograficas directas excluidas:
  - `zona_id`
  - `fact_zona_id`
  - `region`
  - `red_region`
  - `tipo_zona`
  - `red_tipo_zona`
- Agregados anadidos: 48 variables `geo_*` y `rel_*`.
- Variables seleccionadas: 231 features.
- Split: mismo test temporal, desde 2025-06-01 hasta 2025-11-01.
- Train usado: 263099 filas.
- Test temporal: 48888 filas.
- Search mode: `none`.
- Modelos:
  - Dummy baseline.
  - Logistic Regression fija con `C=0.1`.
  - XGBoost con hiperparametros ganadores de Iteracion 2.

Archivos generados:

- `data/modeling/feature_selection_report_it_3b.csv`
- `data/modeling/feature_selection_summary_it_3b.json`
- `reports/models/model_metrics_it_3b.csv`
- `reports/models/model_ranking_it_3b.csv`
- `reports/models/training_summary_it_3b.json`
- `reports/models/test_scores_best_model_it_3b.csv`
- `models/best_model_it_3b.joblib`

### Resultado Iteracion 3B

Modelo ganador automatico: `logistic_regression`, por mayor PR-AUC en test temporal.

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0227 | 0.5278 | 0.0243 | 0.6963 |
| XGBoost | 0.0187 | 0.3849 | 0.0319 | 0.6866 |
| Dummy | 0.0052 | 0.0000 | 0.0000 | 0.5000 |

### Comparacion Iteracion 2 vs 3A vs 3B

| Modelo | PR-AUC it_2 | PR-AUC it_3A | PR-AUC it_3B |
|---|---:|---:|---:|
| Logistic Regression | 0.0237 | 0.0238 | 0.0227 |
| XGBoost | 0.0232 | 0.0219 | 0.0187 |

La Iteracion 3B no mejora a 3A. En Logistic, el PR-AUC baja de 0.0238 a 0.0227. En XGBoost, la caida es mayor: de 0.0219 a 0.0187.

### Criterio 90-95%

Referencia Logistic Iteracion 2: PR-AUC 0.0237.

- 90% de referencia: 0.0213.
- 95% de referencia: 0.0225.
- Logistic Iteracion 3B: 0.0227.

La Iteracion 3B mantiene mas del 95% del PR-AUC de Logistic respecto a Iteracion 2, pero no supera a 3A. Por tanto, no justifica anadir 48 variables agregadas si el objetivo principal es simplificar.

### Overfitting

| Modelo | PR-AUC train | PR-AUC CV | PR-AUC test |
|---|---:|---:|---:|
| Logistic Regression | 0.0202 | 0.0170 | 0.0227 |
| XGBoost | 0.0265 | 0.0180 | 0.0187 |

No se observa overfitting grave, pero XGBoost pierde generalizacion temporal respecto a 3A. Los agregados geograficos no aportan mejora estable con los hiperparametros actuales.

### Lectura Critica Iteracion 3B

La geografia agregada no identificativa no mejora el rendimiento frente a 3A. Esto es una senal importante: el modelo simplificado basado en facturacion, soporte y red ya captura casi toda la senal util sin necesidad de reintroducir contexto geografico adicional.

Logistic detecta 133 de 252 churners, con recall 0.5278, pero genera 10583 falsos positivos. XGBoost detecta 97 churners, con recall 0.3849, y reduce falsos positivos a 5723. XGBoost sigue siendo mas conservador y tiene mejor F1, pero su PR-AUC cae demasiado respecto a Iteracion 2 y 3A.

El resultado tambien sugiere que los agregados por zona/region pueden introducir ruido o redundancia frente a las variables `red_`, `fact_` y `soporte_` ya presentes. Aunque no son identificadores directos, pueden actuar como proxies de contexto geografico sin aportar suficiente valor incremental.

### Decision

No adoptar Iteracion 3B como dataset principal. Mantener Iteracion 3A como baseline simplificado preferido.

Siguientes pasos recomendados:

- Usar 3A para analisis de interpretabilidad y ajuste de threshold.
- Generar feature importance especifica de 3A.
- Si se quiere insistir con geografia, probar una 3B mas pequena con menos agregados y solo variables relativas claramente interpretables.
- Evitar volver a introducir demasiadas variables de contexto si no mejoran PR-AUC o reducen falsos positivos de forma clara.

## Experimento 006 - Iteracion 3C Top 75 Dentro De 3A

- Fecha: 2026-06-17.
- Dataset reducido: `data/modeling/churn_modeling_dataset_it_3c_top75.csv`.
- Dataset base: `data/modeling/churn_modeling_dataset_it_3a.csv`.
- Target: `churn_t_plus_1`.
- Objetivo: comprobar si las top 75 variables de 3A mantienen rendimiento sin cambiar hiperparametros.
- Fuente de seleccion: `reports/models/top_features_75_it_3a.csv`.
- Regla de seleccion: top 75 features originales agrupadas por importancia absoluta del modelo ganador de 3A.
- Variables de entrada 3A: 183 features.
- Variables seleccionadas 3C: 75 features.
- Columnas totales dataset 3C: 78 (`cliente_id`, `fecha`, target y 75 features).
- Split: mismo test temporal, desde 2025-06-01 hasta 2025-11-01.
- Train usado: 263099 filas.
- Test temporal: 48888 filas.
- Search mode: `none`.
- Modelos:
  - Dummy baseline.
  - Logistic Regression fija con `C=0.1`.
  - XGBoost con hiperparametros ganadores de Iteracion 2.

Archivos generados:

- `reports/models/feature_importance_it_3a.csv`
- `reports/models/feature_importance_grouped_it_3a.csv`
- `reports/models/top_features_75_it_3a.csv`
- `data/modeling/feature_selection_report_it_3c_top75.csv`
- `data/modeling/feature_selection_summary_it_3c_top75.json`
- `reports/models/model_metrics_it_3c_top75.csv`
- `reports/models/model_ranking_it_3c_top75.csv`
- `reports/models/training_summary_it_3c_top75.json`
- `reports/models/test_scores_best_model_it_3c_top75.csv`
- `models/best_model_it_3c_top75.joblib`

### Resultado Iteracion 3C

Modelo ganador automatico: `logistic_regression`, por mayor PR-AUC en test temporal.

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0264 | 0.5079 | 0.0251 | 0.7085 |
| XGBoost | 0.0222 | 0.3889 | 0.0305 | 0.6947 |
| Dummy | 0.0052 | 0.0000 | 0.0000 | 0.5000 |

### Comparacion Iteraciones

| Modelo | PR-AUC it_2 | PR-AUC it_3A | PR-AUC it_3B | PR-AUC it_3C Top75 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0237 | 0.0238 | 0.0227 | 0.0264 |
| XGBoost | 0.0232 | 0.0219 | 0.0187 | 0.0222 |

La Iteracion 3C es el mejor resultado hasta ahora para Logistic Regression. Reduce el dataset desde 183 features en 3A hasta 75 features y aumenta el PR-AUC en test temporal.

### Criterio 90-95%

Referencia Logistic Iteracion 3A: PR-AUC 0.0238.

- 90% de referencia: 0.0214.
- 95% de referencia: 0.0226.
- Logistic Iteracion 3C Top75: 0.0264.

3C no solo mantiene el rendimiento de 3A, sino que lo mejora. Por tanto, la reduccion Top 75 es exitosa y debe considerarse el nuevo baseline simplificado principal.

### Overfitting

| Modelo | PR-AUC train | PR-AUC CV | PR-AUC test |
|---|---:|---:|---:|
| Logistic Regression | 0.0204 | 0.0188 | 0.0264 |
| XGBoost | 0.0254 | 0.0178 | 0.0222 |

No se observa overfitting grave. Logistic tiene una diferencia pequena entre train y CV y mejora en test temporal. XGBoost conserva una brecha mayor entre train y CV, aunque no se desploma en test.

### Lectura Critica Iteracion 3C

La mejora de Logistic al reducir a Top 75 sugiere que parte de las variables de 3A aportaban ruido o redundancia. El modelo lineal se beneficia especialmente de una representacion mas compacta.

Matiz sobre geografia: aunque 3C deriva de 3A y no contiene identificadores directos de zona, si conserva algunos proxies de contexto geografico, especialmente `red_poblacion_zona_lag_2m` y `red_poblacion_zona_lag_3m`. Esto no invalida el experimento, pero impide afirmar que 3C esta completamente libre de senal geografica.

XGBoost mejora frente a 3B y queda cerca de 3A, pero no alcanza el rendimiento de Iteracion 2. Aun asi, mantiene mejor F1 que Logistic: 0.0305 frente a 0.0251, porque genera menos falsos positivos.

Confusiones principales en test temporal:

- Logistic Regression: 128 churners detectados, 124 falsos negativos, 9816 falsos positivos.
- XGBoost: 98 churners detectados, 154 falsos negativos, 6067 falsos positivos.

Logistic es mejor si se prioriza detectar mas churners y maximizar PR-AUC. XGBoost es mas conservador si el coste de falsos positivos importa mas.

### Top Variables Del Dataset 3C

Primeras variables seleccionadas:

| Rank | Variable |
|---:|---|
| 1 | `soporte_canal_principal` |
| 2 | `fact_stress_calidad_lag_lag_2m` |
| 3 | `fact_importe_total_lag_1m` |
| 4 | `fact_cargo_base_lag_2m` |
| 5 | `fact_importe_total` |
| 6 | `red_indice_calidad_global_lag_2m` |
| 7 | `soporte_motivo_principal` |
| 8 | `red_indice_calidad_global_lag_3m` |
| 9 | `fact_stress_calidad_lag_roll_3m` |
| 10 | `fact_dias_retraso_pago_roll_3m` |
| 11 | `red_cobertura_5g_pct_lag_2m` |
| 12 | `fact_stress_calidad_lag_lag_1m` |
| 13 | `red_poblacion_zona_lag_2m` *(proxy geografico: revisar)* |
| 14 | `fact_num_lineas_lag_1m` |
| 15 | `fact_importe_total_roll_3m` |

### Decision

Adoptar Iteracion 3C Top75 como nuevo baseline simplificado preferido.

Siguientes pasos recomendados:

- Ajustar threshold sobre Logistic y XGBoost usando `test_scores_best_model_it_3c_top75.csv`.
- Comparar precision/recall por deciles de score.
- Probar una variante `3C_strict` excluyendo `red_poblacion_zona*` para medir cuanto aportan estos proxies geograficos.
- Considerar una prueba secundaria `3C Top75 tuned` solo con XGBoost si se quiere mejorar F1 o reducir falsos positivos.
- Mantener 3C como dataset principal para explicabilidad y documentacion de negocio.

## Experimento 007 - Iteracion 4 Evaluacion Operativa: Threshold Y Lift

- Fecha: 2026-06-17.
- Modelo evaluado: mejor modelo de Iteracion 3C Top75.
- Scores usados: `reports/models/test_scores_best_model_it_3c_top75.csv`.
- Objetivo: evaluar utilidad operativa del modelo para priorizacion de clientes, sin reentrenar.
- Dataset de evaluacion: test temporal 2025-06-01 a 2025-11-01.
- Clientes evaluados: 48888.
- Churners observados: 252.
- Tasa base de churn: 0.5155%.

Archivos generados:

- `reports/models/threshold_optimization_it_4.csv`
- `reports/models/lift_topk_it_4.csv`
- `reports/models/operational_summary_it_4.json`

### Threshold Optimization

El mejor threshold por F1 fue:

| Threshold | Contactados | Churners capturados | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.85 | 229 | 19 | 0.0830 | 0.0754 | 0.0790 |

Este threshold es muy conservador: mejora mucho la precision respecto al umbral 0.5, pero captura pocos churners.

El menor volumen de contacto con recall igual o superior al 50% fue:

| Threshold | Contactados | Churners capturados | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 0.51 | 9309 | 127 | 0.0136 | 0.5040 | 0.0266 |

Este punto es util si el objetivo principal es capturar al menos la mitad de los churners, aunque implica contactar aproximadamente el 19% del test temporal.

### Lift Y Top-K

Ademas de las metricas tradicionales de clasificacion, se evaluo la utilidad operativa del modelo mediante analisis de lift y segmentacion por percentiles de riesgo. Los resultados muestran que el modelo es capaz de concentrar clientes con alta probabilidad de abandono en los segmentos superiores del ranking de riesgo.

| Segmento | Clientes contactados | Churners capturados | Capture rate | Precision@K | Lift |
|---|---:|---:|---:|---:|---:|
| Top 1% | 489 | 22 | 0.0873 | 0.0450 | 8.73 |
| Top 2% | 978 | 39 | 0.1548 | 0.0399 | 7.74 |
| Top 5% | 2444 | 68 | 0.2698 | 0.0278 | 5.40 |
| Top 10% | 4889 | 95 | 0.3770 | 0.0194 | 3.77 |
| Top 15% | 7333 | 116 | 0.4603 | 0.0158 | 3.07 |
| Top 20% | 9778 | 128 | 0.5079 | 0.0131 | 2.54 |
| Top 30% | 14666 | 154 | 0.6111 | 0.0105 | 2.04 |

En el 1% de clientes con mayor score de churn, la tasa de abandono observada fue del 4.50%, frente al 0.52% de la poblacion total, lo que supone un lift de 8.73 veces respecto a una seleccion aleatoria. De forma similar, el 10% de clientes con mayor riesgo concentro aproximadamente el 37.7% de todos los churners observados en el periodo de test.

Estos resultados indican que el modelo puede utilizarse como herramienta de priorizacion para campanas de retencion, permitiendo focalizar recursos comerciales sobre subconjuntos reducidos de clientes con una probabilidad significativamente superior de abandono.

Desde una perspectiva operativa, la estrategia basada en ranking de riesgo y seleccion Top-K resulta mas util que el uso de un umbral fijo de clasificacion, ya que permite adaptar el volumen de clientes contactados a la capacidad y presupuesto disponibles en cada campana.

### Decision

Usar ranking Top-K como enfoque operativo principal. Para una primera campana:

- Si hay poca capacidad comercial: Top 5%, 2444 clientes, captura 68 churners.
- Si se busca mas cobertura: Top 10%, 4889 clientes, captura 95 churners.
- Si el objetivo minimo es capturar la mitad de churners esperados: Top 20% o threshold 0.51.

La siguiente iteracion deberia analizar rendimiento por segmentos de negocio y estimar coste-beneficio de contactar clientes frente al valor esperado de retencion.

## Experimento 008 - Iteracion 5 Politica Operativa Y Coste-Beneficio

- Fecha: 2026-06-17.
- Modelo base: mejor modelo de Iteracion 3C Top75.
- Scores usados: `reports/models/test_scores_best_model_it_3c_top75.csv`.
- Objetivo: traducir el ranking de riesgo a politicas accionables de campana.
- Enfoque: Top-K, lift y escenarios de coste-beneficio.
- No se reentrena ningun modelo.

Archivos generados:

- `reports/models/campaign_policy_it_5.csv`
- `reports/models/cost_benefit_scenarios_it_5.csv`
- `reports/models/operational_recommendation_it_5.json`

### Politicas Top-K

| Politica | Contactados | Churners capturados | Falsos positivos | Precision@K | Capture rate | Lift |
|---|---:|---:|---:|---:|---:|---:|
| Top 1% | 489 | 22 | 467 | 0.0450 | 0.0873 | 8.73 |
| Top 2% | 978 | 39 | 939 | 0.0399 | 0.1548 | 7.74 |
| Top 5% | 2444 | 68 | 2376 | 0.0278 | 0.2698 | 5.40 |
| Top 10% | 4889 | 95 | 4794 | 0.0194 | 0.3770 | 3.77 |
| Top 15% | 7333 | 116 | 7217 | 0.0158 | 0.4603 | 3.07 |
| Top 20% | 9778 | 128 | 9650 | 0.0131 | 0.5079 | 2.54 |
| Top 30% | 14666 | 154 | 14512 | 0.0105 | 0.6111 | 2.04 |

### Escenarios De Coste-Beneficio

Se probaron combinaciones de:

- Coste por contacto: 1, 3 y 5 unidades.
- Valor de retencion por cliente salvado: 50, 100 y 200 unidades.
- Tasa de exito de retencion: 5%, 10% y 20%.

Mejor escenario por valor neto:

| Politica | Contactados | Churners capturados | Coste contacto | Valor retencion | Save rate | Valor neto | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top 2% | 978 | 39 | 1.0 | 200.0 | 0.20 | 582.0 | 0.5951 |

Mejor escenario por ROI positivo:

| Politica | Contactados | Churners capturados | Coste contacto | Valor retencion | Save rate | Valor neto | ROI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Top 1% | 489 | 22 | 1.0 | 200.0 | 0.20 | 391.0 | 0.7996 |

### Lectura Critica Iteracion 5

La politica Top-K confirma que el modelo es util como sistema de priorizacion, no como clasificador binario puro. El Top 1% tiene el mayor lift, pero captura solo 22 churners. El Top 20% captura algo mas de la mitad de churners, pero exige contactar 9778 clientes y genera 9650 falsos positivos.

El coste-beneficio es muy sensible a los supuestos. Con coste de contacto bajo, valor de retencion alto y tasa de exito del 20%, Top 1% y Top 2% pueden ser rentables. Si el coste de contacto sube o la tasa real de retencion baja, la rentabilidad puede desaparecer rapidamente.

La recomendacion operativa por defecto queda en Top 10%, porque captura 95 churners, aproximadamente el 37.7% del total, sin llegar al volumen alto de Top 20%. Aun asi, Top 5% puede ser mas razonable si la capacidad comercial es limitada o si se quiere una campana piloto.

### Decision

Usar Iteracion 5 para definir politica de campana:

- Campana piloto conservadora: Top 5%.
- Campana equilibrada recomendada: Top 10%.
- Campana de alta cobertura: Top 20%, solo si negocio acepta mucho volumen y muchos falsos positivos.

Antes de una decision final real, se deben sustituir los supuestos genericos por datos de negocio:

- coste real por contacto,
- valor esperado de cliente retenido,
- tasa historica de exito de acciones de retencion,
- capacidad mensual del equipo comercial.

## Experimento 009 - Iteracion 6 XGBoost Tuning Avanzado

- Fecha: 2026-06-17.
- Dataset: `data/modeling/churn_modeling_dataset_it_3c_top75.csv`.
- Objetivo: mejorar XGBoost mediante busqueda de hiperparametros sobre el dataset simplificado Top75.
- Comparacion justa: contra Iteracion 3C Top75, porque usa las mismas variables.
- Search mode: `randomized`.
- Iteraciones: 30.
- CV interna: 3 folds.
- Aceleracion solicitada: XGBoost con `device="cuda"` y `tree_method="hist"`.
- Tiempo total de ejecucion: 589.9 segundos, 9.83 minutos.

Archivos generados:

- `reports/models/model_metrics_it_6_xgb_tuned.csv`
- `reports/models/model_ranking_it_6_xgb_tuned.csv`
- `reports/models/model_tuning_results_it_6_xgb_tuned.csv`
- `reports/models/test_scores_best_model_it_6_xgb_tuned.csv`
- `reports/models/training_summary_it_6_xgb_tuned.json`
- `reports/models/runtime_it_6_xgb_tuned.json`
- `models/best_model_it_6_xgb_tuned.joblib`

### Resultado Iteracion 6

Modelo ganador automatico: `logistic_regression`, por mayor PR-AUC en test temporal.

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0264 | 0.5079 | 0.0251 | 0.7085 |
| XGBoost tuned | 0.0229 | 0.4643 | 0.0286 | 0.7075 |
| Dummy | 0.0052 | 0.0000 | 0.0000 | 0.5000 |

### Hiperparametros Ganadores XGBoost

Mejor score CV de XGBoost: PR-AUC 0.0201.

```text
n_estimators=150
max_depth=2
learning_rate=0.03
subsample=1.0
colsample_bytree=0.65
min_child_weight=8
gamma=0.5
reg_alpha=2
reg_lambda=3
```

La configuracion ganadora es mas regularizada y poco profunda. Esto es coherente con el comportamiento observado: XGBoost necesitaba controlar complejidad para no perder generalizacion temporal.

### Comparacion XGBoost: 3C Fijo vs 6 Tuneado

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| XGBoost 3C fijo | 0.0222 | 0.3889 | 0.0305 | 0.6947 |
| XGBoost 6 tuned | 0.0229 | 0.4643 | 0.0286 | 0.7075 |

El tuning mejora PR-AUC, recall y ROC-AUC de XGBoost. Sin embargo, reduce ligeramente F1, porque aumenta la cobertura de churners a costa de mas falsos positivos.

Matriz conceptual en test:

- XGBoost 3C fijo: 98 churners detectados, 154 falsos negativos, 6067 falsos positivos.
- XGBoost 6 tuned: 117 churners detectados, 135 falsos negativos, 7815 falsos positivos.

### Comparacion Frente A Logistic

Logistic sigue siendo el mejor modelo por PR-AUC:

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| Logistic 3C/6 | 0.0264 | 0.5079 | 0.0251 | 0.7085 |
| XGBoost tuned | 0.0229 | 0.4643 | 0.0286 | 0.7075 |

XGBoost tuned mejora el equilibrio F1 frente a Logistic, pero no supera su PR-AUC ni su recall. Logistic sigue siendo el candidato principal si se prioriza ranking global y captura de churners. XGBoost tuned puede ser interesante si se prioriza un comportamiento algo mas conservador con mejor F1.

### Overfitting

| Modelo | PR-AUC train | PR-AUC CV | PR-AUC test |
|---|---:|---:|---:|
| Logistic Regression | 0.0204 | 0.0188 | 0.0264 |
| XGBoost tuned | 0.0226 | 0.0201 | 0.0229 |

No se observa overfitting grave en XGBoost tuned. La diferencia entre train y CV es pequena, y el test temporal queda por encima del CV.

### Decision

No sustituir Logistic como modelo principal. Mantener Logistic 3C Top75 como baseline principal por PR-AUC y simplicidad.

Guardar XGBoost tuned como modelo alternativo:

- mejor que XGBoost fijo en PR-AUC, recall y ROC-AUC,
- mas regularizado,
- potencialmente util si negocio prioriza F1 o menor agresividad que Logistic.

Siguiente paso recomendado: comparar Logistic 3C y XGBoost tuned en metricas operativas Top-K, especialmente Lift@5%, Lift@10%, Precision@K y coste-beneficio. No decidir solo por PR-AUC.

## Experimento 010 - Iteracion 7 Features Manuales

- Fecha: 2026-06-17.
- Dataset base: `data/modeling/churn_modeling_dataset_it_3c_top75.csv`.
- Dataset fuente auxiliar: `data/modeling/churn_modeling_dataset_it_3a.csv`.
- Objetivo: comprobar si unas pocas variables manuales, interpretables y sin leakage mejoran el modelo simplificado 3C.
- Tuning: desactivado inicialmente. Se mantienen los mismos hiperparametros que 3C para aislar el efecto de las variables nuevas.

Variables nuevas:

- `caida_facturacion_3m`: `fact_importe_total_lag_3m - fact_importe_total`.
- `ratio_facturacion_3m`: `fact_importe_total / fact_importe_total_lag_3m`.
- `deterioro_calidad_3m`: `red_indice_calidad_global_lag_3m - red_indice_calidad_global`.
- `tickets_ultimos_90d`: `soporte_contactos + soporte_contactos_lag_1m + soporte_contactos_lag_2m`.

Control de leakage:

- Las variables usan informacion de `t`, `t-1`, `t-2` o `t-3`.
- No usan `churn_t_plus_1`.
- No miran meses posteriores a `t`.
- `tickets_ultimos_90d` se calcula como suma real de contactos y no como `soporte_contactos_roll_3m`, porque el rolling existente es una media movil.

Archivos preparados:

- `src/features/build_dataset_it_7_manual_features.py`
- `src/models/train_models_it_7.py`
- `data/modeling/churn_modeling_dataset_it_7_manual_features.csv`
- `data/modeling/manual_features_report_it_7.csv`
- `data/modeling/manual_features_summary_it_7.json`

Nota de sesgo:

Aunque `genero` no esta presente en los datasets 3A ni 3C actuales, se deja como decision metodologica para la siguiente iteracion: auditar y excluir `genero` si aparece en cualquier dataset de entrenamiento, porque puede introducir sesgo de genero en el modelo.

### Resultado Iteracion 7

Modelo ganador automatico: `logistic_regression`, por mayor PR-AUC en test temporal.

| Modelo | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.0259 | 0.5198 | 0.0249 | 0.7066 |
| XGBoost | 0.0225 | 0.4008 | 0.0301 | 0.6917 |
| Dummy | 0.0052 | 0.0000 | 0.0000 | 0.5000 |

Comparacion directa contra Iteracion 3C Top75:

| Modelo | Iteracion | PR-AUC test | Recall test | F1 test | ROC-AUC test |
|---|---|---:|---:|---:|---:|
| Logistic Regression | 3C Top75 | 0.0264 | 0.5079 | 0.0251 | 0.7085 |
| Logistic Regression | 7 Manual features | 0.0259 | 0.5198 | 0.0249 | 0.7066 |
| XGBoost | 3C Top75 | 0.0222 | 0.3889 | 0.0305 | 0.6947 |
| XGBoost | 7 Manual features | 0.0225 | 0.4008 | 0.0301 | 0.6917 |

Matriz conceptual en test:

- Logistic Regression: 131 churners detectados, 121 falsos negativos, 10123 falsos positivos.
- XGBoost: 101 churners detectados, 151 falsos negativos, 6365 falsos positivos.

### Lectura Critica

La Iteracion 7 no mejora de forma suficiente el modelo principal.

En Logistic Regression, las variables manuales aumentan ligeramente el recall frente a 3C, de 0.5079 a 0.5198, pero empeoran PR-AUC, F1 y ROC-AUC. Como la metrica principal es PR-AUC y el objetivo operativo se basa en ranking de riesgo, esta perdida pesa mas que la pequena ganancia de recall con umbral fijo 0.5.

En XGBoost, PR-AUC y recall suben ligeramente frente a 3C, pero F1 y ROC-AUC bajan. La mejora es demasiado pequena para justificar adoptar el nuevo dataset como principal.

La interpretacion mas probable es que estas variables manuales son razonables desde negocio, pero no anaden informacion claramente nueva: parte de la senal ya estaba capturada por lags, rollings y variables de facturacion, red y soporte presentes en 3C.

### Decision

No adoptar Iteracion 7 como dataset principal.

Mantener `churn_modeling_dataset_it_3c_top75.csv` como baseline simplificado principal. Conservar `churn_modeling_dataset_it_7_manual_features.csv` como experimento documentado y potencial fuente de variables interpretables para explicacion de negocio, pero no como version ganadora.

Siguiente paso recomendado: auditar variables sensibles, especialmente `genero` si aparece en futuros datasets, y ejecutar una iteracion sin variables potencialmente sensibles o proxies claros. Despues, priorizar evaluacion operativa Top-K y calibracion/threshold antes de seguir anadiendo variables manuales.

## Experimento 011 - Iteracion 3D Top50

- Fecha: 2026-06-17.
- Dataset base: `data/modeling/churn_modeling_dataset_it_3a.csv`.
- Fuente de importancia: `reports/models/top_features_75_it_3a.csv`.
- Objetivo: volver a la linea simplificada anterior y comprobar si las 50 variables mas importantes mantienen el rendimiento de 3C Top75.
- Tuning: desactivado. Se mantienen los mismos hiperparametros que 3C para que la comparacion mida solo el efecto de reducir variables.

Archivos preparados:

- `src/features/build_reduced_dataset_it_3d_top50.py`
- `src/models/train_models_it_3d_top50.py`
- `data/modeling/churn_modeling_dataset_it_3d_top50.csv`
- `data/modeling/feature_selection_report_it_3d_top50.csv`
- `data/modeling/feature_selection_summary_it_3d_top50.json`

Criterio de lectura:

- Comparar principalmente contra 3C Top75.
- Si PR-AUC cae poco y el modelo mantiene lift operativo razonable, Top50 puede ser preferible por simplicidad.
- Si PR-AUC o recall caen demasiado, mantener 3C Top75.
- Revisar de nuevo proxies geograficos, especialmente variables tipo `red_poblacion_zona_*`, antes de adoptar Top50 como version final.

### Resultado Iteracion 3E

Modelo ganador automatico: `logistic_regression`, por mayor PR-AUC en test temporal.

| Modelo | PR-AUC test | ROC-AUC test | Recall test | Precision test | F1 test |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.0241 | 0.7120 | 0.4960 | 0.0142 | 0.0276 |
| XGBoost | 0.0221 | 0.6993 | 0.3968 | 0.0167 | 0.0321 |
| Dummy | 0.0052 | 0.5000 | 0.0000 | 0.0000 | 0.0000 |

Comparativa automatica:

| Iteracion | Features | Modelo ganador | PR-AUC test | ROC-AUC test | Recall test | Precision test | F1 test |
|---|---:|---|---:|---:|---:|---:|---:|
| 3C Top75 | 75 | Logistic Regression | 0.0264 | 0.7085 | 0.5079 | 0.0129 | 0.0251 |
| 3D Top50 | 50 | Logistic Regression | 0.0266 | 0.7130 | 0.5317 | 0.0131 | 0.0256 |
| 3E Top30 | 30 | Logistic Regression | 0.0241 | 0.7120 | 0.4960 | 0.0142 | 0.0276 |

### Overfitting / Underfitting

| Modelo | Gap train-CV PR-AUC | Gap train-test PR-AUC | Gap CV-test PR-AUC | Diagnostico |
|---|---:|---:|---:|---|
| Dummy | 0.0000 | 0.0014 | 0.0014 | underfitting |
| Logistic Regression | 0.0005 | -0.0037 | -0.0042 | ok |
| XGBoost | 0.0052 | 0.0022 | -0.0030 | ok |

No se observa overfitting grave en Logistic ni en XGBoost. El problema de 3E no es sobreajuste, sino perdida de senal al reducir de 50 a 30 variables.

### Multicolinealidad

El analisis de correlacion y VIF muestra multicolinealidad fuerte incluso en Top30.

Variables con VIF mas alto:

| Variable | VIF |
|---|---:|
| `fact_num_lineas` | inf |
| `fact_num_lineas_roll_3m` | inf |
| `fact_importe_total_roll_3m` | 458.02 |
| `fact_importe_total` | 387.33 |
| `fact_cargo_base_lag_3m` | 135.41 |
| `fact_importe_total_lag_3m` | 105.86 |
| `fact_cargo_base_lag_1m` | 80.80 |
| `fact_cargo_base_lag_2m` | 66.12 |

Pares con correlacion absoluta extrema:

- `fact_num_lineas_roll_3m` vs `fact_num_lineas`: 1.0000.
- `fact_num_lineas_lag_1m` vs `fact_num_lineas`: 1.0000.
- `fact_cargo_base_lag_3m` vs `fact_cargo_base_lag_1m`: 1.0000.
- `fact_cargo_base_lag_2m` vs `fact_cargo_base_lag_3m`: 1.0000.
- `red_poblacion_zona_lag_3m` vs `red_poblacion_zona_lag_2m`: 1.0000.
- `fact_importe_total` vs `fact_importe_total_roll_3m`: 0.9985.

Tambien persisten proxies geograficos:

- `red_poblacion_zona_lag_3m`
- `red_poblacion_zona_lag_2m`

### Decision

No adoptar Iteracion 3E Top30 como modelo principal.

Aunque Top30 mejora precision y F1 frente a 3D, empeora la metrica principal PR-AUC y reduce el recall. Como el objetivo principal es ranking de riesgo y deteccion de churners, esa perdida no compensa la reduccion adicional de variables.

La mejor version simplificada sigue siendo Iteracion 3D Top50:

- mejor PR-AUC test,
- mejor ROC-AUC test,
- mejor recall,
- reduccion relevante frente a 3C,
- sin evidencia de overfitting grave.

Siguiente paso recomendado: no seguir bajando variables solo por ranking de importancia. Si se busca simplificar mas, hacerlo eliminando redundancias por grupos correlacionados: elegir una variable representante por familia (`importe_total`, `cargo_base`, `num_lineas`, `calidad_global`, etc.) y quitar proxies geograficos como `red_poblacion_zona_*`.

## Experimento 013 - Iteracion 3F Poda Por Familias

- Fecha: 2026-06-17.
- Dataset base: `data/modeling/churn_modeling_dataset_it_3d_top50.csv`.
- Objetivo: reducir multicolinealidad sin seguir bajando variables por Top-N.
- Tuning: desactivado. Se mantienen los mismos hiperparametros que 3D.

Regla aplicada:

- Mantener `fact_importe_total_roll_3m` como representante de importe.
- Eliminar `fact_importe_total`, `fact_importe_total_lag_1m`, `fact_importe_total_lag_2m`, `fact_importe_total_lag_3m`.
- Mantener `fact_cargo_base_lag_1m` como representante de cargo base.
- Eliminar `fact_cargo_base_lag_2m`, `fact_cargo_base_lag_3m`.
- Mantener `fact_num_lineas` como representante de lineas.
- Eliminar `fact_num_lineas_lag_1m`, `fact_num_lineas_roll_3m`.
- Mantener `red_indice_calidad_global_roll_3m` como representante de calidad de red.
- Eliminar `red_indice_calidad_global_lag_1m`, `red_indice_calidad_global_lag_2m`, `red_indice_calidad_global_lag_3m`.
- Eliminar completamente proxies geograficos `red_poblacion_zona_lag_2m` y `red_poblacion_zona_lag_3m`.

Archivos preparados:

- `src/features/build_reduced_dataset_it_3f_family_pruning.py`
- `src/models/train_models_it_3f_family_pruning.py`
- `data/modeling/churn_modeling_dataset_it_3f_family_pruning.csv`
- `data/modeling/feature_selection_report_it_3f_family_pruning.csv`
- `data/modeling/feature_selection_summary_it_3f_family_pruning.json`

Resultados pendientes de ejecucion.

## Experimento 014 - Iteracion 3G Logistic Con Imputacion Semantica

- Fecha: 2026-06-17.
- Dataset base: `data/modeling/churn_modeling_dataset_it_3f_family_pruning.csv`.
- Objetivo: probar si Logistic Regression mejora con imputacion mas semantica frente a la imputacion generica mediana/moda.
- Modelos: Dummy y Logistic Regression.
- Tuning: desactivado.

Reglas de imputacion:

- Mantener `add_indicator=True` en numericas.
- Imputar a 0 variables numericas de eventos, conteos y flags: soporte, contactos, incidencias, impagos, duraciones de soporte sin actividad, flags de missing y conteos de encuestas.
- Imputar con mediana el resto de numericas: importes, calidad, retrasos y variables continuas.
- Imputar categoricas con `"Desconocido"` en vez de moda.
- Mantener escalado con `StandardScaler` para Logistic Regression.
- Mantener one-hot para categoricas.

Archivos preparados:

- `src/models/train_models_it_3g_logistic_semantic_imputation.py`
- `reports/models/model_metrics_it_3g_logistic_semantic_imputation.csv`
- `reports/models/model_ranking_it_3g_logistic_semantic_imputation.csv`
- `reports/models/test_scores_best_model_it_3g_logistic_semantic_imputation.csv`
- `models/best_model_it_3g_logistic_semantic_imputation.joblib`

Resultados pendientes de ejecucion.

## Experimento 012 - Iteracion 3E Top30 + Multicolinealidad

- Fecha: 2026-06-17.
- Dataset base: `data/modeling/churn_modeling_dataset_it_3d_top50.csv`.
- Fuente de importancia: importancia real del modelo ganador de 3D Top50.
- Objetivo: reducir de Top50 a Top30 variables y anadir diagnostico de multicolinealidad.
- Tuning: desactivado. Se mantienen los mismos hiperparametros que 3D.

Reglas de columnas:

- `cliente_id`, `fecha` y `churn_t_plus_1` se mantienen en el CSV exportado.
- `cliente_id` y `fecha` no entran al entrenamiento.
- `churn_t_plus_1` solo se usa como variable objetivo.
- Solo las Top30 variables entran como `X_train` y `X_test`.

Archivos preparados:

- `src/models/export_feature_importance_it_3d_top50.py`
- `src/features/build_reduced_dataset_it_3e_top30.py`
- `src/models/train_models_it_3e_top30.py`
- `data/modeling/churn_modeling_dataset_it_3e_top30.csv`
- `data/modeling/feature_selection_report_it_3e_top30.csv`
- `data/modeling/feature_selection_summary_it_3e_top30.json`
- `reports/models/feature_importance_it_3e_top30.csv`
- `reports/models/feature_importance_grouped_it_3e_top30.csv`
- `reports/models/feature_correlation_matrix_it_3e_top30.csv`
- `reports/models/feature_vif_it_3e_top30.csv`
- `reports/models/high_correlation_pairs_it_3e_top30.csv`
- `reports/models/overfitting_diagnostics_it_3e_top30.csv`
- `reports/models/iteration_comparison_it_3e_top30.csv`
- `reports/figures/correlation_heatmap_it_3e_top30.png`

Nota metodologica:

El analisis de correlacion y VIF se calcula solo sobre variables numericas Top30. Las variables categoricas no se incluyen en VIF para evitar interpretaciones artificiales derivadas del one-hot encoding.

Resultados pendientes de ejecucion.
