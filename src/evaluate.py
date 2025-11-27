import joblib
from pathlib import Path

from sklearn.metrics import classification_report, roc_auc_score

from data_prep import load_data, split_data


def main():
    # Cargar datos y separar train/test
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Cargar pipeline entrenado
    base_dir = Path(__file__).resolve().parents[1]
    model_path = base_dir / "api" / "model_pipeline.pkl"

    pipe = joblib.load(model_path)

    # Predicciones y probabilidades
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred))

    print("ROC-AUC:", roc_auc_score(y_test, y_proba))


if __name__ == "__main__":
    main()