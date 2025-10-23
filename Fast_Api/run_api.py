from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os
import json
import uvicorn
model = None  # глобально, но не загружаем сразу
# === Загрузка модели ===


# === Инициализация FastAPI ===
app = FastAPI(
    title="Sleep Quality Prediction API",
    description="API для предсказания качества сна: 0 = bad, 1 = good, 2 = medium",
    version="1.0"
)


@app.on_event("startup")
def load_model():
    global model
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "RandomForest_Sleep.pkl")
    model = joblib.load(model_path)
    print(f"✅ Модель загружена из: {model_path}")


# === Схема входных данных ===
class SleepData(BaseModel):
    Age: float
    Gender: int
    Sleep_duration: float
    Awakenings: float
    Caffeine_consumption: float
    Alcohol_consumption: float
    Smoking_status: int
    Exercise_frequency: float
    bed_hour: float
    wake_hour: float


# === Главная страница ===
@app.get("/")
def root():
    return {"message": "Отправь POST-запрос на /predict для предсказания качества сна."}


# === Эндпоинт предсказания ===
@app.post("/predict")
def predict(data: SleepData):
    X = np.array([[getattr(data, field) for field in data.__fields__]])
    y_pred = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        confidence = float(np.max(probs))
        return {
            "sleep_efficiency_label": int(y_pred),
            "sleep_quality": ["bad", "good", "medium"][int(y_pred)],
            "confidence": round(confidence, 3)
        }

    return {
        "sleep_quality_label": int(y_pred),
        "sleep_quality": ["bad", "good", "medium"][int(y_pred)]
    }


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 8080))
    uvicorn.run("run_api:app", host="0.0.0.0", port=port)
