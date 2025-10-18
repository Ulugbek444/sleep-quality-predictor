import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from ml_experiments.config.experiment_config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, ML_FLOW_MODEL_NAME
from ml_experiments.utils.mlflow_setup import setup_mlflow
from ml_experiments.utils.staging_manager import auto_stage_best_model
from ml_experiments.report_manager.model_registry import (list_model_versions, load_model_version,
                                                          tag_register_model, compare_multiple_models,
                                                          list_models_by_model_stage_tag,
                                                          delete_model_versions_by_stage)
from mlflow.tracking import MlflowClient


def registry():

    setup_mlflow(MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI)
    print(f"Запуск эксперимента: {MLFLOW_EXPERIMENT_NAME}")
    client = MlflowClient()

    # models = client.search_registered_models()
    #
    # for model in models:
    #     print(f"Удаление модели: {model.name}")
    #     client.delete_registered_model(name=model.name)

    # TODO: Описаение к модели по версии
    # client.update_model_version(
    #     name="LogisticRegression_LungCancer",
    #     version="2",  # номер версии
    #     description="score = f1_weighted"
    # )

    # TODO: Для выбора и отметки самой лучшей модели по показателям
    # auto_stage_best_model(f"KNeighborsClassifier_{ML_FLOW_MODEL_NAME}", metric_tags=["f1_score_test", "roc_auc_test"])
    # auto_stage_best_model(f"GaussianNB_{ML_FLOW_MODEL_NAME}", metric_tags=["f1_score_test", "roc_auc_test"])
    # auto_stage_best_model(f"LogisticRegression_{ML_FLOW_MODEL_NAME}", metric_tags=["f1_score_test", "roc_auc_test"])
    # auto_stage_best_model(f"RandomForest_{ML_FLOW_MODEL_NAME}", metric_tags=["f1_score_test", "roc_auc_test"])
    # auto_stage_best_model(f"XGBoost_{ML_FLOW_MODEL_NAME}", metric_tags=["f1_score_test", "roc_auc_test"])

    # TODO: Для вывода всех моделей по регистру
    # list_model_versions("XGBoost_LungCancer", sort_by="version", descending=False)
    # TODO: Для получения конкретной версии модели из Model Registry
    model = load_model_version(model_name=f"XGBoost_{ML_FLOW_MODEL_NAME}", version=5)
    save_dir = r"/Fast_Api/models"
    os.makedirs(save_dir, exist_ok=True)  # создаст папку, если её нет
    file_path = os.path.join(save_dir, f"XGBoost_{ML_FLOW_MODEL_NAME}.pkl")
    joblib.dump(model, file_path)
    print(f"Модель сохранена по пути: {file_path}")
    # TODO: Для перевода модели на указанную стадию (Production, Staging, Archived, None, To_Delete)
    # Добавление тега к любой модели зная версию, по сути можно задать любой тег (key) + значение (value)
    # tag_register_model(model_name="KNeighborsClassifier_LungCancer", version=1, key="model_stage", value="To_Delete")
    # TODO: Для сравнения 2 или более моделей ("model_name", version)
    # compare_multiple_models([
    #     ("XGBoost_LungCancer", 9),
    #     ("LogisticRegression_LungCancer", 4)
    # ])
    # compare_multiple_models([
    #     ("XGBoost_LungCancer", 9),
    #     ("XGBoost_LungCancer", 10)],
    #     metrics_to_compare=["f1_score_test", "roc_auc_test"]
    # )
    # compare_multiple_models([
    #     ("XGBoost_LungCancer", 9),
    #     ("RandomForest_LungCancer", 2),
    #     ("LogisticRegression_LungCancer", 4)
    # ])
    # TODO: Вывод моделей по model_stage (Production, Staging, Archived, None, To_Delete)
    # list_models_by_model_stage_tag("Staging")
    # TODO: Удаление моделей по stage, можно задать в stage tag_register_model например To_Delete затем этой
    #  функцией удалить, dry_run = True только показывает какая удалиться а False удалит
    # delete_model_versions_by_stage(stage_filter="To_Delete", dry_run=False)
    # TODO: Вывод тегов конкретной модели
    # version_info = client.get_model_version("KNeighborsClassifier_LungCancer", version="1")
    # print(version_info.tags)


if __name__ == "__main__":
    registry()
