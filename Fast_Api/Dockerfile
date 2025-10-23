FROM python:3.12-slim

WORKDIR /app

# Необходимые системные библиотеки
RUN apt-get update && apt-get install -y build-essential gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements_local.txt

COPY . .

CMD ["sh", "-c", "uvicorn run_api:app --host 0.0.0.0 --port ${PORT:-8080}"]
