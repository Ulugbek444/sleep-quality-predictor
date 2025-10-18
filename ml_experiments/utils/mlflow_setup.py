import os
from dotenv import load_dotenv
# Абсолютный путь к ../.env
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_USER = os.getenv("MLFLOW_USER")

import mlflow
import mlflow.sklearn
from ml_experiments.config.experiment_config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MLFLOW_USER


def setup_mlflow(experiment_name=MLFLOW_EXPERIMENT_NAME, tracking_uri=MLFLOW_TRACKING_URI):
    """Настраивает MLflow с Model Registry"""
    print("🔍 Проверка переменных:")
    print("MLFLOW_EXPERIMENT_NAME:", MLFLOW_EXPERIMENT_NAME)
    print("MLFLOW_TRACKING_URI:", MLFLOW_TRACKING_URI)
    print("MLFLOW_USER:", MLFLOW_USER)
    # Имя пользователя для логов / MLflow
    os.environ['USER'] = os.getenv("MLFLOW_USER", "anonymous")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    mlflow.sklearn.autolog(disable=True)

    print(f"🔧 MLflow настроен:")
    print(f"   Experiment: {experiment_name}")
    print(f"   Tracking URI: {tracking_uri}")
    print(f"   Model Registry: включен")

    return mlflow
