import json
import asyncio
import logging
from aiohttp import ClientSession
from bot.questions import questions
from bot.utils import convert_to_12_hour, format_answers_for_api, validate_sleep_data, generate_advice
from bot.session import user_data, user_results
from config.token import API_TOKEN, FASTAPI_URL
from aiogram import Router, types, F
from aiogram.filters import Command
from aiogram import Bot

router = Router()


@router.message(Command("start"))
async def start(message: types.Message):
    text = (
        "👋 Привет! Я **SleepAdvisorBot**, помогу оценить качество твоего сна.\n\n"
        "🧠 /check — пройти опрос и узнать качество сна\n"
        "💡 /help — советы по улучшению сна\n\n"
        "Выбери команду, чтобы начать!"
    )
    await message.answer(text, parse_mode="Markdown")


@router.message(Command("help"))
async def help_command(message: types.Message):
    user_id = message.from_user.id
    result = user_results.get(user_id)

    if not result:
        text = (
            f"🧠 Ты ещё не прошёл анализ сна!\n\n"
            "💡 Но всё равно держи базовые советы для улучшения сна:\n"
            "- Ложись и вставай в одно и то же время.\n"
            "- Избегай кофеина и алкоголя перед сном.\n"
            "- Не смотри в экран за час до сна.\n"
            "- Проветривай комнату и держи прохладу.\n"
            "- Делай лёгкую физическую активность днём.\n\n"
            "Попробуй /check, чтобы узнать качество сна!"
        )
    else:
        text = (
            f"🧠 Ты уже прошёл анализ сна!\n\n"
            f"У тебя {result['label']}\n\n"
            f"{result['advice']}"
        )

    await message.answer(text, parse_mode="HTML")



@router.message(Command("check"))
async def check_start(message: types.Message):
    user_id = message.from_user.id
    user_data[user_id] = {"step": 0, "answers": {}, "message": None}

    sent = await message.answer("🧾 Начинаем опрос...")
    user_data[user_id]["message"] = sent
    await asyncio.sleep(1)
    await ask_question(user_id, message.bot)


async def ask_question(user_id: int, bot: Bot):
    try:
        step = user_data[user_id]["step"]
        msg = user_data[user_id]["message"]
    except KeyError:
        logging.warning(f"Пользователь {user_id} не найден в user_data")
        await bot.send_message(user_id, "⚠️ Сессия устарела. Попробуй /check заново.")
        return

    # Завершение опроса и отправка данных в API
    if step >= len(questions):
        await bot.send_message(user_id, "🔄 Анализирую твой сон...")
        await asyncio.sleep(1)
        answers = user_data[user_id]["answers"]
        payload = format_answers_for_api(answers)

        if not validate_sleep_data(payload):
            await bot.send_message(user_id, "⚠️ Некоторые значения некорректны. Попробуй /check заново.")
            user_data.pop(user_id, None)
            return

        try:
            async with ClientSession() as session:
                async with session.post(FASTAPI_URL, json=payload, timeout=10) as resp:
                    if resp.status == 200:
                        try:
                            result = await resp.json()
                            label = result.get("sleep_efficiency_label", -1)
                            label_map = {
                                0: "❌ Плохой сон",
                                1: "✅ Хороший сон",
                                2: "😐 Средний сон"
                            }
                            label_text = label_map.get(label, "неизвестно")
                            # Сохраняем результат и советы
                            user_data[user_id]["last_label"] = label_text
                            user_data[user_id]["last_advice"] = generate_advice(user_data[user_id]["answers"], label)

                            # Отправляем сообщение
                            await bot.send_message(
                                user_id,
                                f"🧠 Предсказание модели: <b>{label_text}</b>\n\n{user_data[user_id]['last_advice']}",
                                parse_mode="HTML"
                            )
                        except json.JSONDecodeError:
                            logging.exception("Ошибка декодирования JSON-ответа API")
                            await bot.send_message(user_id, "⚠️ Сервер вернул некорректный ответ.")
                    else:
                        await bot.send_message(user_id, f"⚠️ Ошибка от API: {resp.status}")
        except asyncio.TimeoutError:
            await bot.send_message(user_id, "⏱️ Превышено время ожидания ответа от сервера.")
        except Exception:
            logging.exception("Ошибка при обращении к API")
            await bot.send_message(user_id, "❌ Произошла ошибка при анализе. Попробуй позже.")

        user_results[user_id] = {
            "label": label_text,
            "advice": generate_advice(user_data[user_id]["answers"], label)
        }
        user_data.pop(user_id, None)
        return

    # Задание вопроса
    try:
        key, text, qtype = questions[step]
        markup = None

        if key == "Gender":
            age = user_data[user_id]["answers"].get("Age", 20)
            options = [("Девушка", 0), ("Юноша", 1)] if age <= 18 else [("Женщина", 0), ("Мужчина", 1)]
            markup = types.InlineKeyboardMarkup(inline_keyboard=[
                [types.InlineKeyboardButton(text=n, callback_data=f"{key}:{v}")] for n, v in options
            ])
        elif key == "Smoking_status":
            markup = types.InlineKeyboardMarkup(inline_keyboard=[
                [types.InlineKeyboardButton(text="🚭 Нет", callback_data=f"{key}:0"),
                 types.InlineKeyboardButton(text="🚬 Да", callback_data=f"{key}:1")]
            ])

        if qtype == "choice" and markup is None:
            logging.warning(f"Нет вариантов для вопроса {key}")
            await bot.send_message(user_id, "⚠️ Ошибка: отсутствуют варианты ответа.")
            return

        sent = await bot.send_message(user_id, text, reply_markup=markup)
        user_data[user_id]["message"] = sent
    except Exception:
        logging.exception("Ошибка при формировании вопроса")
        await bot.send_message(user_id, "❌ Произошла ошибка. Попробуй /check заново.")
        user_data.pop(user_id, None)


@router.message(F.text)
async def handle_text(message: types.Message):
    user_id = message.from_user.id
    if user_id not in user_data:
        return

    step = user_data[user_id]["step"]
    key, _, qtype = questions[step]

    if qtype != "input":
        return

    raw = message.text.strip()

    # ⏳ Небольшая пауза перед обработкой
    await asyncio.sleep(0.5)

    # 🔍 Проверка на корректный ввод
    try:
        if key in ["bed_hour", "wake_hour"]:
            val = convert_to_12_hour(raw)
        else:
            # Проверка на чистое число
            if not raw.replace(".", "", 1).isdigit():
                raise ValueError("Невалидный ввод")
            val = float(raw)
    except ValueError:
        await message.answer("❌ Неправильный ввод. Попробуйте ещё раз.")
        return

    # 🔧 Специфические проверки
    if key == 'Exercise_frequency':
        if val < 0:
            await message.answer("Понял, значит не занимаетесь)")
            val = 0
        elif val > 7:
            await message.answer("В неделе только 7 дней! 😅 Попробуйте указать реальное количество.")
            return

    if key == "Caffeine_consumption":
        if val < 0:
            await message.answer("Понял, значит не пьёте)")
            val = 0
        val = min(val * 25, 125)

    if key == 'Alcohol_consumption':
        if val < 0:
            await message.answer("Понял, значит не пьёте)")
            val = 0
        elif val > 10:
            await message.answer("Мда...")

    if key == 'Awakenings':
        if val > 10:
            await message.answer("Эмм, вы вообще спали? Введите ещё раз:")
            return

    if key == "Sleep_duration":
        if val > 24:
            await message.answer("В сутках только 24 часа! 😅 Попробуйте указать реальное количество сна.")
            return

    if key == "Age":
        if "age_attempts" not in user_data[user_id]:
            user_data[user_id]["age_attempts"] = 0

        if val < 0:
            user_data[user_id]["age_attempts"] += 1
            if user_data[user_id]["age_attempts"] == 1:
                await message.answer("Возраст не бывает отрицательным так-то...\nПопробуйте ещё раз:")
                return
            elif user_data[user_id]["age_attempts"] == 2:
                await message.answer("Эмм...\nВы странный 😅 Может ещё раз)?")
                return
            elif user_data[user_id]["age_attempts"] >= 3:
                await message.answer("Так уж и быть... Окей, записываю...")
                val = 20

        elif 100 <= val <= 120:
            await message.answer("Вот это вы долгожитель! 🎉")

        elif val > 120:
            user_data[user_id]["pending_age"] = val
            markup = types.InlineKeyboardMarkup(inline_keyboard=[
                [
                    types.InlineKeyboardButton(text="Да", callback_data="AgeConfirm:yes"),
                    types.InlineKeyboardButton(text="Нет", callback_data="AgeConfirm:no")
                ]
            ])
            await message.answer(
                "Максимальный документально подтверждённый возраст человека — 122 года и 164 дня.\n"
                "Этот рекорд принадлежит француженке Жанне Кальман.\n"
                "Вы уверены, что вам больше неё?",
                reply_markup=markup
            )
            return

    # ✅ Сохраняем и переходим к следующему шагу
    user_data[user_id]["answers"][key] = val
    user_data[user_id]["step"] += 1

    await ask_question(user_id, message.bot)


@router.callback_query(F.data)
async def handle_callback(call: types.CallbackQuery):
    user_id = call.from_user.id
    if user_id not in user_data:
        await call.answer()
        return

    try:
        # 🔍 Обработка подтверждения возраста
        if call.data.startswith("AgeConfirm"):
            choice = call.data.split(":")[1]
            if "pending_age" not in user_data[user_id]:
                await call.answer("Сессия устарела. Попробуй /check заново.")
                return

            await call.message.delete()

            if choice == "yes":
                val = user_data[user_id]["pending_age"]
                await call.message.answer(f"Хорошо, записываю возраст как {val}")
                user_data[user_id]["answers"]["Age"] = val
                user_data[user_id]["step"] += 1
                user_data[user_id].pop("pending_age", None)
                await ask_question(user_id, call.bot)

            elif choice == "no":
                await call.message.answer("Хорошо, тогда введите другой возраст:")
                user_data[user_id].pop("pending_age", None)

            await call.answer()
            return  # ⛔️ Не продолжаем дальше

        # 🔁 Обычная обработка кнопок
        key, value = call.data.split(":")

        await call.message.delete()

        user_data[user_id]["answers"][key] = int(value)
        user_data[user_id]["step"] += 1
        await call.answer()
        await ask_question(user_id, call.bot)

    except Exception:
        logging.exception("Ошибка в callback")
        await call.answer("⚠️ Ошибка обработки ответа.")
        await call.bot.send_message(user_id, "❌ Что-то пошло не так. Попробуй /check заново.")
        user_data.pop(user_id, None)
