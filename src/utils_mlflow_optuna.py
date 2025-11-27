import mlflow

EXPERIMENT_NAME = "taller3-heart-risk"


def init_mlflow():
    """
    Inicializa el experimento de MLflow.
    Usa el tracking URI por defecto (local) a menos que se reconfigure.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)