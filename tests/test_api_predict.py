import json
import pytest
import requests

# === Загрузка тестовых данных ===
with open("tests/Json_test_samples/api_test_features_collinearity.json") as f:
    features = json.load(f)

with open("tests/Json_test_samples/api_test_labels_collinearity.json") as f:
    labels = json.load(f)

API_URL = "http://127.0.0.1:8000/predict"


# === Счётчики для метрик
results = {
    "correct": 0,
    "total": len(features),
    "errors": []
}


# === Параметризованный тест ===
@pytest.mark.parametrize("sample,expected", zip(features, labels))
def test_prediction(sample, expected):
    response = requests.post(API_URL, json=sample)
    assert response.status_code == 200

    result = response.json()
    predicted = result["sleep_quality_label"]
    print(f"🔍 predicted={predicted}, expected={expected}")

    if predicted == expected:
        results["correct"] += 1
    else:
        results["errors"].append((sample, expected, predicted))

    assert predicted == expected


# === Финальный отчёт после всех тестов ===
def test_summary():
    accuracy = results["correct"] / results["total"]
    print(f"\n📊 Accuracy: {accuracy:.2%} ({results['correct']}/{results['total']})")
    print(f"\n✅ Модель прошла {results['correct']} из {results['total']} тестов")
    print(f"📊 Accuracy: {accuracy:.2%}")
    if results["errors"]:
        print(f"⚠️ Ошибок: {len(results['errors'])}")
        for i, (sample, expected, predicted) in enumerate(results["errors"]):
            print(f"  [{i+1}] expected={expected}, predicted={predicted}, input={sample}")
