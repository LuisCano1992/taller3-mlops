import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from config import TARGET, RANDOM_STATE, TEST_SIZE


def load_data(path: str = "data/raw/heart.csv") -> pd.DataFrame:
    """
    Carga el dataset desde la ruta indicada.
    """
    return pd.read_csv(path)


def split_data(df: pd.DataFrame):
    """
    Separa el DataFrame en train/test estratificado según la variable objetivo.

    Retorna:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Crea un preprocesador que estandariza todas las columnas numéricas.
    En este dataset todas las features son numéricas.
    """
    numeric_features = df.drop(columns=[TARGET]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ]
    )
    return preprocessor