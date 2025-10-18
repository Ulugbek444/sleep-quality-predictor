import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
# Абсолютный путь к ../.env
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
from ml_experiments.config.experiment_config import (
    PROCESSED_DATA_PATH_WITHOUT_COLLINEARITY,
    PROCESSED_DATA_PATH_COLLINEARITY,
    PROCESSED_DATA_PATH_WITHOUT_COLLINEARITY_FORXGBOOST,
    PROCESSED_DATA_PATH_WITH_COLLINEARITY_FOR_XG_RF_NO_REM,
    PROCESSED_DATA_PATH_NO_COLLINEARITY_NO_REM,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
    MLFLOW_MODEL_NAME
)
from ml_experiments.utils.preprocessing import oversample_dataset
from ml_experiments.report_manager.model_registry import load_model_version


def load_data(oversample=False, samples=10, save_test_samples=False):
    df = pd.read_csv(PROCESSED_DATA_PATH_WITH_COLLINEARITY_FOR_XG_RF_NO_REM)

    # X = df.drop(columns=["sleep_efficiency_label"])
    # print("🔍 Длина X.columns:", len(X.columns))
    # model = load_model_version(model_name=f"XGBoost_{MLFLOW_MODEL_NAME}", version=5)
    # print(model.named_steps)
    # rf_model = model.named_steps["model"]
    # feature_importances = rf_model.feature_importances_
    # print("🔍 Длина feature_importances_:", len(feature_importances))
    # features = X.columns
    # importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importances}).sort_values(by="Importance",
    #                                                                                                    ascending=False)
    # print(importance_df)
    # Сначала отделяем тест
    df_temp, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["sleep_efficiency_label"],
        random_state=RANDOM_STATE
    )

    # Затем делим оставшееся на train и valid
    valid_ratio = VALIDATION_SIZE / (1 - TEST_SIZE)  # пересчитанная доля от оставшегося
    train_df, valid_df = train_test_split(
        df_temp,
        test_size=valid_ratio,
        stratify=df_temp["sleep_efficiency_label"],
        random_state=RANDOM_STATE
    )

    # Разделяем признаки и метки
    x_train = train_df.drop("sleep_efficiency_label", axis=1).values
    y_train = train_df["sleep_efficiency_label"].values
    x_valid = valid_df.drop("sleep_efficiency_label", axis=1).values
    y_valid = valid_df["sleep_efficiency_label"].values
    x_test = test_df.drop("sleep_efficiency_label", axis=1).values
    y_test = test_df["sleep_efficiency_label"].values

    # Oversampling
    x_train, y_train = oversample_dataset(x_train, y_train, oversample=oversample)
    x_valid, y_valid = oversample_dataset(x_valid, y_valid, oversample=oversample)
    x_test, y_test = oversample_dataset(x_test, y_test, oversample=oversample)
    if save_test_samples:
        n_samples = samples
        feature_names = test_df.drop("sleep_efficiency_label", axis=1).columns.tolist()

        # Сохраняем features
        features = [
            dict(zip(feature_names, x_test[i]))
            for i in range(min(n_samples, len(x_test)))
        ]

        # Сохраняем labels — в том же порядке
        labels = [int(y_test[i]) for i in range(min(n_samples, len(y_test)))]

        # Построение пути на 2 уровня вверх
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        features_path = os.path.join(base_dir, "tests", "Json_test_samples", "api_test_features_no_collinearity.json")
        labels_path = os.path.join(base_dir, "tests", "Json_test_samples", "api_test_labels_no_collinearity.json")

        # Убедимся, что папка существует
        os.makedirs(os.path.dirname(features_path), exist_ok=True)
        os.makedirs(os.path.dirname(labels_path), exist_ok=True)
        # Сохраняем оба файла
        with open(features_path, "w") as f:
            json.dump(features, f, indent=2)

        with open(labels_path, "w") as f:
            json.dump(labels, f, indent=2)

        print(f"✅ Сохранено {len(features)} features в {features_path}")
        print(f"✅ Сохранено {len(labels)} labels в {labels_path}")

    return x_train, y_train, x_valid, y_valid, x_test, y_test
