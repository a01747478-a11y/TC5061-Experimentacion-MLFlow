
import argparse
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def parse_args():
    """Recibe los argumentos desde la terminal."""
    parser = argparse.ArgumentParser(
        description="Experimentos de Wine Quality con MLflow"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="winequality-red.csv"
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=10
    )

    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--tracking_uri",
        type=str,
        default=None
    )

    return parser.parse_args()


def load_data(data_path):
    """Carga el dataset y elimina duplicados."""
    df = pd.read_csv(data_path, sep=";")
    df = df.drop_duplicates().copy()
    return df


def prepare_data(df, seed):
    """Separa las variables de entrenamiento y prueba."""
    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=seed
    )

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, args):
    """Entrena el modelo Random Forest."""
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=args.seed,
        n_jobs=1
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Calcula las métricas y devuelve las predicciones."""
    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        ),
        "R2": r2_score(y_test, predictions)
    }

    return metrics, predictions


def save_plot(y_test, predictions, output_path):
    """Crea la gráfica de valores reales y predicciones."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        y_test,
        predictions,
        alpha=0.6
    )

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        label="Predicción perfecta"
    )

    ax.set_xlabel("Calidad real")
    ax.set_ylabel("Calidad predicha")
    ax.set_title("Valores reales vs. predicciones")
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    np.random.seed(args.seed)


    # Configurar MLflow usando una base de datos SQLite
    if args.tracking_uri:
        mlflow.set_tracking_uri(args.tracking_uri)
    else:
        db_path = Path("mlflow.db").resolve()
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")


    # Agrupar todas las corridas en un experimento
    mlflow.set_experiment("WineQuality_RandomForest")

    # Comenzar una corrida de MLflow
    with mlflow.start_run() as run:

        # Registrar los hiperparámetros
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param(
            "min_samples_split",
            args.min_samples_split
        )
        mlflow.log_param("seed", args.seed)

        # Preparar datos
        df = load_data(args.data_path)

        X_train, X_test, y_train, y_test = prepare_data(
            df, args.seed
        )

        # Entrenar modelo
        model = train_model(X_train, y_train, args)

        # Evaluar modelo
        metrics, predictions = evaluate_model(
            model, X_test, y_test
        )

        # Registrar métricas
        for name, value in metrics.items():
            mlflow.log_metric(name, float(value))

        # Crear y guardar una gráfica como artefacto
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_path = Path(temp_dir) / "predicciones.png"

            save_plot(y_test, predictions, plot_path)

            mlflow.log_artifact(
                str(plot_path),
                artifact_path="graficas"
            )


        # Guardar el modelo entrenado
        mlflow.sklearn.log_model(
          sk_model=model,
          name="random_forest_model",
          input_example=X_train.head(3),
          serialization_format="cloudpickle"
        )


        print("\n=== RESULTADOS DEL MODELO ===")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")

        print("\n=== INFORMACIÓN DE MLFLOW ===")
        print("Experimento: WineQuality_RandomForest")
        print("Run ID:", run.info.run_id)
        print("Tracking URI:", mlflow.get_tracking_uri())

        print("\nCorrida registrada correctamente.")


if __name__ == "__main__":
    main()
