from mlflow.tracking import MlflowClient


def auto_stage_best_model(model_name: str, metric_tags: list = ["f1_score_test"], strategy: str = "max"):
    """
    Находит лучшую версию модели по заданным метрикам и присваивает ей алиас 'staging'.

    Args:
        list = ["f1_score_test"]: Если не задается то по дефолту
        model_name (str): Имя модели в MLflow Model Registry.
        metric_tags (list): Список тегов метрик, по которым выбирается лучшая модель.
        strategy (str): 'max' (по умолчанию) или 'min' — направление оптимизации.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")

    best_version = None
    best_score = None

    for v in versions:
        try:
            scores = []
            for tag in metric_tags:
                val = v.tags.get(tag)
                if val is None:
                    raise ValueError(f"Нет тега '{tag}' у версии {v.version}")
                scores.append(float(val))

            # Среднее значение всех метрик
            avg_score = sum(scores) / len(scores)

            if best_score is None or \
               (strategy == "max" and avg_score > best_score) or \
               (strategy == "min" and avg_score < best_score):
                best_score = avg_score
                best_version = v

        except Exception as e:
            print(f"⚠️ Пропущена версия {v.version}: {e}")

    if best_version:
        client.set_registered_model_alias(
            name=model_name,
            version=best_version.version,
            alias="staging"
        )
        print(f"✅ '{model_name}' версия {best_version.version} установлена как 'staging' (среднее по {metric_tags} = {best_score:.4f})")
    else:
        print(f"❌ Не удалось найти подходящую версию модели '{model_name}'")
