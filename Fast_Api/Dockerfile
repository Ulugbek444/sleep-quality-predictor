FROM python:3.12-slim

WORKDIR /app

# Устанавливаем необходимые системные библиотеки
RUN apt-get update && apt-get install -y build-essential gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код FastAPI
COPY . .

# Запуск FastAPI
CMD ["uvicorn", "run_api:app", "--host", "0.0.0.0", "--port", "8080"]
