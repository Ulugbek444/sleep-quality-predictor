import os
from dotenv import load_dotenv
# Абсолютный путь к ../.env
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
from ml_experiments.config.experiment_config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from ml_experiments.utils.mlflow_setup import setup_mlflow
from ml_experiments.models.LogisticRegression import logistic_regression_experiment
from ml_experiments.models.KNN import knn_experiment
from ml_experiments.models.naive_bayes import naive_bayes_experiment
from ml_experiments.models.xgboost import xgboost_experiment
from ml_experiments.models.random_forest import random_forest_experiment
from ml_experiments.utils.data_processing import load_data


def main():

    # TODO: Флаг oversample
    use_oversample = False

    # TODO: Настройка MLflow
    setup_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)
    print(f"Запуск эксперимента: {MLFLOW_EXPERIMENT_NAME}")

    # TODO: Загрузка данных, samples кол-во данных для теста в API, save_test_samples чтобы только 1 раз сохранять
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(oversample=use_oversample,
                                                                   samples=10, save_test_samples=False)
    # LG_model = logistic_regression_experiment(x_train, y_train, x_valid, y_valid, x_test, y_test,
    #                                           oversample=use_oversample)
    # KNN_model = knn_experiment(x_train, y_train, x_valid, y_valid, x_test, y_test,
    #                            oversample=use_oversample)
    # NB = naive_bayes_experiment(x_train, y_train, x_valid, y_valid, x_test, y_test,
    #                             oversample=use_oversample)
    # xg_model = xgboost_experiment(x_train, y_train, x_valid, y_valid, x_test, y_test,
    #                               oversample=use_oversample)
    RF = random_forest_experiment(x_train, y_train, x_valid, y_valid, x_test, y_test,
                                  oversample=use_oversample)


if __name__ == "__main__":
    main()

