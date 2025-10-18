import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from collections import defaultdict

# === ФУНКЦИИ ДЛЯ УПРАВЛЕНИЯ ВЕРСИЯМИ МОДЕЛЕЙ ===


def load_model_version(model_name, version=None, stage=None):
    """
    Загружает конкретную версию модели из Model Registry

    Args:
        model_name: Имя модели в реестре
        version: Номер версии (например, "1", "2")
        stage: Стадия модели ("Staging", "Production", "Archived")
    """
    try:
        if version:
            model_uri = f"models:/{model_name}/{version}"
            print(f"Загружаем модель {model_name} версии {version}")
        elif stage:
            model_uri = f"models:/{model_name}/{stage}"
            print(f"Загружаем модель {model_name} со стадии {stage}")
        else:
            model_uri = f"models:/{model_name}/latest"
            print(f"Загружаем последнюю версию модели {model_name}")

        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Модель успешно загружена: {model_uri}")
        return model

    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {e}")
        return None


def list_model_versions(model_name, sort_by="f1_score_test", descending=True):
    """Показывает все версии модели, отсортированные по заданному полю."""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")

        rows = []
        for v in versions:
            tags = v.tags
            row = {
                "version": int(v.version),  # <-- это настоящий номер версии
                "stage": v.current_stage,
                "created": v.creation_timestamp,
                "run_id": v.run_id,
                "tags": tags,
                "f1_score_test": float(tags.get("f1_score_test", -1)),
                "roc_auc_test": float(tags.get("roc_auc_test", -1))
            }
            rows.append(row)

        # Выбор поля сортировки
        if sort_by == "version":
            rows_sorted = sorted(rows, key=lambda x: x["version"], reverse=descending)
        elif sort_by == "created":
            rows_sorted = sorted(rows, key=lambda x: x["created"], reverse=descending)
        elif sort_by in ["f1_score_test", "roc_auc_test"]:
            rows_sorted = sorted(rows, key=lambda x: x.get(sort_by, -1), reverse=descending)
        else:
            print(f"⚠️ Неизвестное поле сортировки: '{sort_by}', сортировка отключена.")
            rows_sorted = rows

        print(f"\n📋 Версии модели '{model_name}' (сортировка по '{sort_by}'):")
        print("-" * 80)
        for row in rows_sorted:
            print(f"Версия: {row['version']}")
            print(f"Стадия: {row['stage']}")
            print(f"Создана: {row['created']}")
            print(f"Run ID: {row['run_id']}")
            print(f"f1_score_test: {row['f1_score_test']}")
            print(f"roc_auc_test: {row['roc_auc_test']}")
            print(f"Теги: {row['tags']}")
            print("-" * 40)

    except Exception as e:
        print(f"❌ Ошибка при получении версий: {e}")


def tag_register_model(model_name, version, key="None", value="None"):
    """
    Добавляет тег к зарегистрированной модели (не к версии).

    Args:
        model_name (str): Имя модели в реестре
        version (str or int): Номер версии модели
        key (str): Название тега
        value (str): Значение тега
    """
    try:
        client = MlflowClient()
        version_str = str(version)
        # Тег к зарегистрированной модели
        client.set_registered_model_tag(
            name=model_name,
            key=key,
            value=value
        )
        # Тег к конкретной версии модели
        client.set_model_version_tag(
            name=model_name,
            version=version_str,
            key=key,
            value=value
        )
        print(f"🏷️ Тег '{key}={value}' добавлен к зарегистрированной модели '{model_name}'")

    except Exception as e:
        print(f"❌ Ошибка при добавлении тега к модели: {e}")


def compare_multiple_models(model_versions, metrics_to_compare=None):
    """
    Сравнивает любое количество моделей по заданным метрикам.

    Args:
        model_versions (list of tuples): Список моделей и версий в формате [(model_name, version), ...]
        metrics_to_compare (list, optional): Метрики для сравнения. Если не указано — используется стандартный набор.
    """
    try:
        client = MlflowClient()

        # Метрики по умолчанию
        if metrics_to_compare is None:
            metrics_to_compare = [
                'f1_score_test',
                'roc_auc_test',
                'accuracy_test',
                'precision_test',
                'recall_test'
            ]

        # Получаем метрики для каждой модели
        results = []
        for model_name, version in model_versions:
            version_str = str(version)
            mv = client.get_model_version(model_name, version_str)
            run = client.get_run(mv.run_id)

            metrics = {}
            for metric in metrics_to_compare:
                value = run.data.metrics.get(metric)
                metrics[metric] = value if value is not None else "❌"

            results.append({
                "model": model_name,
                "version": version_str,
                "metrics": metrics
            })

        # Выводим таблицу сравнения
        print("\n📊 Сравнение моделей:")
        print("=" * 80)
        header = f"{'Model':30} | " + " | ".join([f"{m:>12}" for m in metrics_to_compare])
        print(header)
        print("-" * len(header))

        for entry in results:
            row = f"{entry['model']} (v{entry['version']})".ljust(30) + " | "
            row += " | ".join([
                f"{entry['metrics'][m]:>12.4f}" if isinstance(entry['metrics'][m], float) else f"{entry['metrics'][m]:>12}"
                for m in metrics_to_compare
            ])
            print(row)

    except Exception as e:
        print(f"❌ Ошибка при сравнении моделей: {e}")


def list_models_by_model_stage_tag(filter_stage=None):
    """
    Выводит зарегистрированные модели и их версии, сгруппированные по тегу 'model_stage',
    установленному через set_model_version_tag.

    Args:
        filter_stage (str, optional): Фильтр по значению тега 'model_stage' (например, 'Production')
    """
    try:
        client = MlflowClient()
        all_models = client.search_registered_models()

        stage_groups = {}

        for model in all_models:
            model_name = model.name
            versions = client.search_model_versions(f"name='{model_name}'")

            for v in versions:
                tags = client.get_model_version(model_name, v.version).tags
                stage = tags.get("model_stage", "None")

                if filter_stage is None or stage == filter_stage:
                    stage_groups.setdefault(stage, []).append((model_name, f"v{v.version}"))

        if not stage_groups:
            print(f"\n🔍 Нет версий моделей с тегом 'model_stage={filter_stage}'")
            return

        print("\n📦 Версии моделей по тегу 'model_stage':")
        print("=" * 50)
        for stage in sorted(stage_groups.keys()):
            entries = stage_groups[stage]
            print(f"\n🔹 {stage} ({len(entries)}):")
            for name, version in entries:
                print(f"   - {name} → {version}")

    except Exception as e:
        print(f"❌ Ошибка при получении моделей: {e}")


def delete_model_versions_by_stage(stage_filter="To_Delete", dry_run=True):
    """
    Удаляет только версии моделей, у которых тег 'model_stage' соответствует stage_filter.

    Args:
        stage_filter (str): Значение тега 'model_stage', по которому фильтруются версии для удаления.
        dry_run (bool): Если True — только показывает, какие версии будут удалены, без фактического удаления.
    """
    try:
        client = MlflowClient()
        all_models = client.search_registered_models()

        to_delete = []

        for model in all_models:
            model_name = model.name
            versions = client.search_model_versions(f"name='{model_name}'")

            for v in versions:
                tags = client.get_model_version(model_name, v.version).tags
                stage = tags.get("model_stage", "None")

                if stage == stage_filter:
                    to_delete.append((model_name, v.version))

        if not to_delete:
            print(f"\n🔍 Нет версий моделей с тегом 'model_stage={stage_filter}' для удаления.")
            return

        print(f"\n🧹 Версии моделей с тегом 'model_stage={stage_filter}':")
        for model_name, version in to_delete:
            print(f"   - {model_name} → v{version}")
            if not dry_run:
                client.delete_model_version(name=model_name, version=str(version))
                print(f"     ✅ Удалено")

        if dry_run:
            print("\nℹ️ dry_run=True → версии не были удалены. Установи dry_run=False для удаления.")

    except Exception as e:
        print(f"❌ Ошибка при удалении версий моделей: {e}")

