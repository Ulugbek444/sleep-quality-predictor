import time
import numpy as np
import mlflow
import mlflow.sklearn
import os
from importlib.metadata import version
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, make_scorer, f1_score, precision_score,
                             recall_score, accuracy_score, classification_report)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from mlflow.tracking import MlflowClient

# Импорты для визуализации
from ml_experiments.utils.visualization import save_confusion_matrix, save_roc_curve, save_precision_recall_curve


def run_experiment(model_name, model_class, run_name,
                   grid_param, x_tr, y_tr, x_vl, y_vl, x_te, y_te,
                   scaler=False, mix=False, register_model=True,
                   model_registry_name=None, refit_metric='f1_weighted', average="weighted"):
    """
    Запускает эксперимент с машинным обучением и версионированием модели

    Args:
        model_name (str): Имя модели.
        model_class (class): Класс модели (например, sklearn GaussianNB).
        run_name (str): Имя запуска MLflow.
        grid_param (dict): Словарь с параметрами для GridSearchCV.
        x_tr (np.array): Признаки для обучения.
        y_tr (np.array): Метки для обучения.
        x_vl (np.array): Признаки для валидации.
        y_vl (np.array): Метки для валидации.
        x_te (np.array): Признаки для теста.
        y_te (np.array): Метки для теста.
        scaler (bool, optional): Применять ли StandardScaler. Default is False.
        mix (bool, optional): Объединять ли train+valid для финального обучения. Default is False.
        register_model (bool, optional): Регистрировать модель в MLflow Model Registry. Default is True.
        model_registry_name (str, optional): Имя модели в реестре. Default is None.
        refit_metric (str): Название метрики, по которой выбирается лучшая модель после GridSearchCV.
        Например 'f1_weighted', 'accuracy', 'roc_auc_ovr_weighted'
        average (str): Метод усреднения метрики для многоклассовой классификации.
        - 'macro': усреднение по всем классам без учёта их частоты
        - 'weighted': усреднение с учётом количества примеров каждого класса
        - 'micro': глобальное усреднение по всем примерам
        Рекомендуется 'weighted' при наличии дисбаланса классов.
    """

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("sklearn_version", version("scikit-learn"))
        mlflow.log_param("timestamp", time.strftime("%Y-%m-%d %H:%M:%S"))

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("train_size", len(x_tr))
        mlflow.log_param("valid_size", len(x_vl))
        mlflow.log_param("test_size", len(x_te))

        steps = []
        if scaler:
            steps.append(('scaler', StandardScaler()))

        steps.append(('model', model_class()))
        pipeline = Pipeline(steps)

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=grid_param,
            scoring=refit_metric,
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(x_tr, y_tr)
        best_model = grid.best_estimator_
        mlflow.log_params(grid.best_params_)

        # Validation
        print("Validation of model...")
        y_valid_pred = best_model.predict(x_vl)
        y_valid_prob = best_model.predict_proba(x_vl)

        metrics_valid = {
            "accuracy_valid": accuracy_score(y_vl, y_valid_pred),
            "precision_valid": precision_score(y_vl, y_valid_pred, average=average),
            "recall_valid": recall_score(y_vl, y_valid_pred, average=average),
            "f1_score_valid": f1_score(y_vl, y_valid_pred, average=average),
            "roc_auc_valid": roc_auc_score(y_vl, y_valid_prob, multi_class='ovr', average=average)
        }
        print("=== Validation Metrics ===")
        print(classification_report(y_vl, y_valid_pred))
        mlflow.log_metrics(metrics_valid)

        save_confusion_matrix(y_vl, y_valid_pred, run_name, "valid")
        save_roc_curve(y_vl, y_valid_prob, run_name, "valid")
        save_precision_recall_curve(y_vl, y_valid_prob, run_name, "valid")

        # Объединение Train + Valid
        if mix:
            print("Переобучение на объединенных данных (train + valid)...")
            x_train_full = np.vstack([x_tr, x_vl])
            y_train_full = np.concatenate([y_tr, y_vl])
            # Создаем новую модель с теми же параметрами
            last_model = Pipeline(steps)
            last_model.set_params(**grid.best_params_)
            last_model.fit(x_train_full, y_train_full)

            mlflow.log_param("train_used", "train+valid")
        else:
            last_model = best_model  # если не объединяем, используем модель как есть
            mlflow.log_param("train_used", "train_only")

        # Final Test
        print("Тестирование финальной модели...")
        y_test_pred = last_model.predict(x_te)
        y_test_prob = last_model.predict_proba(x_te)

        metrics_test = {
            "accuracy_test": accuracy_score(y_te, y_test_pred),
            "precision_test": precision_score(y_te, y_test_pred, average=average),
            "recall_test": recall_score(y_te, y_test_pred, average=average),
            "f1_score_test": f1_score(y_te, y_test_pred, average=average),
            "roc_auc_test": roc_auc_score(y_te, y_test_prob, multi_class='ovr', average=average)
        }
        print("=== Test Metrics ===")
        print(classification_report(y_te, y_test_pred))
        mlflow.log_metrics(metrics_test)

        save_confusion_matrix(y_te, y_test_pred, run_name, "test")
        save_roc_curve(y_te, y_test_prob, run_name, "test")
        save_precision_recall_curve(y_te, y_test_prob, run_name, "test")

        # Сохраняем модель как артефакт
        signature = infer_signature(x_tr, last_model.predict(x_tr))
        mlflow.sklearn.log_model(
            sk_model=last_model,
            name="model",
            signature=signature,
            input_example=x_tr[:5],
        )

        # === ВЕРСИОНИРОВАНИЕ МОДЕЛИ ===
        if register_model:
            try:
                # Получаем текущий run_id
                run_id = mlflow.active_run().info.run_id

                # Определяем имя модели в реестре
                if model_registry_name is None:
                    model_registry_name = f"{model_name}_LungCancer"

                # Регистрируем модель в Model Registry
                f1_value = float(metrics_test['f1_score_test'])
                roc_value = float(metrics_test['roc_auc_test'])

                model_uri = f"runs:/{run_id}/model"
                model_version = mlflow.register_model(
                    model_uri=model_uri,
                    name=model_registry_name,
                    tags={
                        "model_type": model_name,
                        "experiment_date": time.strftime("%Y-%m-%d"),
                        "data_preprocessing": "scaler" if scaler else "no_scaler",
                        "training_strategy": "train+valid" if mix else "train_only",
                        "f1_score_test": f"{f1_value:.4f}",
                        "roc_auc_test": f"{roc_value:.4f}",
                        "model_stage": "Staging"
                    }
                )

                print(f"✅ Модель зарегистрирована в Model Registry:")
                print(f"   Имя: {model_registry_name}")
                print(f"   Версия: {model_version.version}")
                print(f"   URI: {model_uri}")

                # Логируем информацию о версии модели
                mlflow.log_param("model_registry_name", model_registry_name)
                mlflow.log_param("model_version", model_version.version)

                # Добавляем тег model_stage к версии и к run
                client = mlflow.tracking.MlflowClient()
                client.set_model_version_tag(
                    name=model_registry_name,
                    version=model_version.version,
                    key="model_stage",
                    value="None"
                )
                client.set_tag(run_id, "model_stage", "None")

                description = (
                    f"Модель: {model_name}; "
                    f"Дата: {time.strftime('%Y-%m-%d')}; "
                    f"Стратегия: {'train+valid' if mix else 'train_only'}; "
                    f"Масштабирование: {'включено' if scaler else 'нет'}; "
                    f"Метрика выбора: {refit_metric}; "
                    f"Усреднение: {average}; "
                    f"F1 (test): {f1_value:.4f}; "
                    f"ROC AUC (test): {roc_value:.4f}"
                )
                client.update_model_version(
                    name=model_registry_name,
                    version=str(model_version.version),
                    description=description
                )

                # Автоматически переводим модель в статус "Staging" если метрики хорошие
                if metrics_test['f1_score_test'] > 0.9 and metrics_test['roc_auc_test'] > 0.9:
                    client.set_registered_model_alias(
                        name=model_registry_name,
                        version=model_version.version,
                        alias="staging"
                    )
                    print(f"🚀 Модель переведена в стадию 'Staging' и получила alias 'staging' (хорошие метрики)")

                return last_model, metrics_valid, metrics_test, model_version

            except Exception as e:
                print(f"⚠️ Ошибка при регистрации модели: {e}")
                print("Модель сохранена как артефакт, но не зарегистрирована в реестре")

        # Логируем данные
        try:
            mlflow.log_artifact("../data/processed_data/survey_lung_cancer_clean.csv")
        except Exception as e:
            print(f"Предупреждение: не удалось залогировать файл данных: {e}")

        print("Эксперимент завершен успешно!")
        return last_model, metrics_valid, metrics_test, None
