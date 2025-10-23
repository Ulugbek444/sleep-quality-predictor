from dotenv import load_dotenv
import os

# Поднимаемся на два уровня вверх до корня проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(BASE_DIR, ".env")

load_dotenv(ENV_PATH)

API_TOKEN = os.getenv("BOT_TOKEN")
FASTAPI_URL = os.getenv("FASTAPI_URL")
