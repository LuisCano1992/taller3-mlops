import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
import optuna

from sklearn.pipeline import Pipeline
from sklearn.metrics import recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from data_prep import load_data, split_data, build_preprocessor
from utils_mlflow_optuna import init_mlflow


# =====================================
# Objetivos de Optuna por modelo
# =====================================

def objective_logreg(trial, X_train, y_train, X_valid, y_valid, preprocessor):
    C = trial.suggest_float("C", 0.0001, 10.0, log=True)

    model = LogisticRegression(C=C, max_iter=1000)

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run(nested=True):
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_valid)

        recall = recall_score(y_valid, pred)
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_metric("recall", recall)

    return recall


def objective_rf(trial, X_train, y_train, X_valid, y_valid, preprocessor):
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run(nested=True):
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_valid)

        recall = recall_score(y_valid, pred)
        mlflow.log_param("model", "RandomForest")
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth
        })
        mlflow.log_metric("recall", recall)

    return recall


def objective_gb(trial, X_train, y_train, X_valid, y_valid, preprocessor):
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.5)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)

    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run(nested=True):
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_valid)

        recall = recall_score(y_valid, pred)
        mlflow.log_param("model", "GradientBoosting")
        mlflow.log_params({
            "learning_rate": learning_rate,
            "n_estimators": n_estimators
        })
        mlflow.log_metric("recall", recall)

    return recall


# =====================================
# MAIN
# =====================================

def main():
    # 1. Inicializar MLflow
    init_mlflow()

    # 2. Cargar datos y separar train/valid
    df = load_data()
    X_train, X_valid, y_train, y_valid = split_data(df)

    # 3. Crear preprocesador
    preprocessor = build_preprocessor(df)

    # 4. Optuna para cada modelo

    # Logistic Regression
    study_lr = optuna.create_study(direction="maximize", study_name="logreg_study")
    study_lr.optimize(
        lambda t: objective_logreg(t, X_train, y_train, X_valid, y_valid, preprocessor),
        n_trials=20
    )
    print("Mejor recall LR:", study_lr.best_value)

    # Random Forest
    study_rf = optuna.create_study(direction="maximize", study_name="rf_study")
    study_rf.optimize(
        lambda t: objective_rf(t, X_train, y_train, X_valid, y_valid, preprocessor),
        n_trials=20
    )
    print("Mejor recall RF:", study_rf.best_value)

    # Gradient Boosting
    study_gb = optuna.create_study(direction="maximize", study_name="gb_study")
    study_gb.optimize(
        lambda t: objective_gb(t, X_train, y_train, X_valid, y_valid, preprocessor),
        n_trials=20
    )
    print("Mejor recall GB:", study_gb.best_value)

    # 5. Comparar recalls
    recalls = {
        "logreg": study_lr.best_value,
        "rf": study_rf.best_value,
        "gb": study_gb.best_value
    }
    print("Recalls por modelo:", recalls)

    best_model_name = max(recalls, key=recalls.get)
    print("Modelo ganador:", best_model_name)

    # 6. Instanciar el mejor modelo con sus mejores hiperparámetros
    if best_model_name == "logreg":
        best_model = LogisticRegression(
            C=study_lr.best_params["C"],
            max_iter=1000
        )
    elif best_model_name == "rf":
        best_model = RandomForestClassifier(
            n_estimators=study_rf.best_params["n_estimators"],
            max_depth=study_rf.best_params["max_depth"],
            random_state=42,
            n_jobs=-1
        )
    else:
        best_model = GradientBoostingClassifier(
            learning_rate=study_gb.best_params["learning_rate"],
            n_estimators=study_gb.best_params["n_estimators"]
        )

    # 7. Pipeline final (prep + modelo ganador)
    final_pipe = Pipeline([
        ("prep", preprocessor),
        ("model", best_model)
    ])

    final_pipe.fit(X_train, y_train)

    # 8. Guardar pipeline en api/model_pipeline.pkl
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "api" / "model_pipeline.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipe, model_path)

    print("Pipeline final guardado en:", model_path)
    print("Tamaño del archivo:", model_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()