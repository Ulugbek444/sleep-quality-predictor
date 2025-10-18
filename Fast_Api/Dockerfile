# 🐍 Базовый образ Python
FROM python:3.12-slim
# 📁 Рабочая директория внутри контейнера
WORKDIR /app
# 📦 Копируем локальный requirements.txt
COPY requirements.txt .
# 📦 Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
# 📁 Копируем всё содержимое Fast_Api
COPY . .
# 🚀 Запуск FastAPI
CMD ["uvicorn", "run_api:app", "--host", "0.0.0.0", "--port", "8080"]