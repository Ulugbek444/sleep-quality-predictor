import logging
import asyncio
import os
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart, Command
from config.token import API_TOKEN, FASTAPI_URL
from bot.handlers import router

# Настройка логирования
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
dp.include_router(router)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(dp.start_polling(bot))
    except Exception as e:
        logging.exception("Фатальная ошибка при запуске бота")
