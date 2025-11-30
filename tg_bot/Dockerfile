FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY tg_bot/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY tg_bot/ .

# 🚀 Запускаем main.py
CMD ["python", "-m", "main"]



