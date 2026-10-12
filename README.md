# Experimentación de Machine Learning con MLflow

## Descripción

Este proyecto implementa un proceso de experimentación reproducible para predecir la calidad del vino tinto utilizando RandomForestRegressor.

El entrenamiento se realiza mediante un script parametrizado en Python. MLflow Tracking registra los hiperparámetros, las métricas, los artefactos y los modelos generados en cada ejecución.

## Dataset

Se utiliza Wine Quality - Red Wine, del repositorio UCI Machine Learning.

- Observaciones originales: 1,599
- Observaciones después de eliminar duplicados: 1,359
- Variables predictoras: 11
- Variable objetivo: quality
- División: 80 % entrenamiento y 20 % prueba

## Requisitos

- Python 3.10
- MLflow 3.5.1
- scikit-learn 1.5.2
- pandas 2.2.3
- NumPy 1.26.4
- Matplotlib 3.9.2

## Instalación

Clonar el repositorio y acceder a la carpeta del proyecto.

Crear un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instalar las dependencias:

```bash
python -m pip install -r requirements.txt
```

## Ejecución

Ejecutar con los hiperparámetros predeterminados:

```bash
python train.py
```

Ejecutar con una configuración personalizada:

```bash
python train.py --n_estimators 50 --max_depth 10 --min_samples_split 2 --seed 42
```

## Hiperparámetros

- `n_estimators`: número de árboles.
- `max_depth`: profundidad máxima.
- `min_samples_split`: mínimo de muestras para dividir un nodo.
- `seed`: semilla aleatoria.

## Experimentos realizados

| Corrida | n_estimators | max_depth | min_samples_split | seed |
|---|---:|---:|---:|---:|
| 1 | 100 | 10 | 2 | 42 |
| 2 | 50 | 10 | 2 | 42 |
| 3 | 200 | 10 | 2 | 42 |
| 4 | 100 | 5 | 2 | 42 |
| 5 | 100 | 10 | 6 | 42 |

Todas las corridas se agrupan en el experimento `WineQuality_RandomForest`.

## Métricas

Se registran tres métricas:

- MAE: Mean Absolute Error.
- RMSE: Root Mean Squared Error.
- R2: coeficiente de determinación.

## Visualización en MLflow

Desde la carpeta del proyecto, ejecutar:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Abrir en el navegador:

http://127.0.0.1:5000

MLflow permite consultar los parámetros, métricas, gráficas y modelos entrenados.

## Reproducibilidad

Se utiliza la semilla aleatoria 42 y se fijan las versiones principales de las dependencias en `requirements.txt`.

Los datos se dividen utilizando `train_test_split` con `random_state=seed`.

El modelo RandomForestRegressor utiliza la misma semilla y `n_jobs=1`.

## Resultados

Entre las cinco configuraciones evaluadas, el mejor desempeño según RMSE se obtuvo con:

- n_estimators: 50
- max_depth: 10
- min_samples_split: 2
- seed: 42

Métricas:

- MAE: 0.4597
- RMSE: 0.6066
- R2: 0.4806

## Autor

Santiago Espinosa

Actividad de experimentación y seguimiento de modelos con MLflow.
