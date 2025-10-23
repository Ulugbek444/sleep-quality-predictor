FROM python:3.12-slim

WORKDIR /app

# Необходимые системные библиотеки
RUN apt-get update && apt-get install -y build-essential gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run_api.py"]

