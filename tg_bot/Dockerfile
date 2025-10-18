# ✅ Используем Python 3.12, как в FastAPI
FROM python:3.12-slim

# 📁 Рабочая директория внутри контейнера
WORKDIR /app

# 📦 Копируем локальный requirements.txt
COPY requirements.txt .

# 📦 Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# 📁 Копируем весь код бота
COPY . .

# 👇 Добавляем /app в PYTHONPATH, чтобы tg_bot стал доступен как пакет
ENV PYTHONPATH=/app

# 🚀 Запускаем main.py
CMD ["python", "-m", "main"]



