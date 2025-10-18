import logging


def convert_to_12_hour(value: str) -> float:
    try:
        # Удаляем пробелы и обрабатываем формат HH:MM
        value = value.strip()
        if ":" in value:
            hour, minute = value.split(":")
            hour = int(hour)
            minute = int(minute)
            val = hour + minute / 60
        else:
            val = float(value)

        val = int(val) % 24  # ограничиваем до 0–23
        return 12 if val == 0 else val % 12 or 12
    except Exception as e:
        logging.warning(f"Ошибка преобразования времени: {value} → {e}")
        return 0.0


def format_answers_for_api(raw: dict) -> dict:
    return {
        "Age": float(raw.get("Age", 0)),
        "Gender": int(raw.get("Gender", 0)),
        "Sleep_duration": float(raw.get("Sleep_duration", 0)),
        "Awakenings": float(raw.get("Awakenings", 0)),
        "Caffeine_consumption": float(raw.get("Caffeine_consumption", 0)),
        "Alcohol_consumption": float(raw.get("Alcohol_consumption", 0)),
        "Smoking_status": int(raw.get("Smoking_status", 0)),
        "Exercise_frequency": float(raw.get("Exercise_frequency", 0)),
        "bed_hour": float(raw.get("bed_hour", 0)),
        "wake_hour": float(raw.get("wake_hour", 0))
    }


def validate_sleep_data(data: dict) -> bool:
    return (
        0 < data["Age"] < 120 and
        0 <= data["Sleep_duration"] <= 24 and
        0 <= data["Awakenings"] <= 20 and
        0 <= data["Caffeine_consumption"] <= 200 and
        0 <= data["Alcohol_consumption"] <= 50 and
        0 <= data["Exercise_frequency"] <= 14 and
        0 < data["bed_hour"] <= 12 and
        0 < data["wake_hour"] <= 12
    )


def generate_advice(data: dict, label: int) -> str:
    """
    Генерация персональных советов по улучшению сна
    на основе введённых пользователем данных и важности признаков.
    """
    advice = ["💡 <b>Персональные рекомендации:</b>"]

    # --- Awakenings ---
    if data.get("Awakenings", 0) > 2:
        advice.append("• Старайтесь сократить количество пробуждений — возможно, стоит проветрить комнату или подобрать удобную подушку.")
    elif data.get("Awakenings", 0) == 0:
        advice.append("• Отлично! Вы спите без пробуждений — сохраняйте этот режим.")

    # --- Alcohol ---
    if data.get("Alcohol_consumption", 0) > 3:
        advice.append("• Алкоголь перед сном снижает качество сна. Постарайтесь не употреблять за 3–4 часа до сна.")
    elif 0 < data.get("Alcohol_consumption", 0) <= 3:
        advice.append("• Даже умеренное употребление алкоголя может влиять на фазы сна.")

    # --- Exercise ---
    if data.get("Exercise_frequency", 0) == 0:
        advice.append("• Добавьте лёгкую физическую активность — прогулку или растяжку в течение дня.")
    elif data.get("Exercise_frequency", 0) > 5:
        advice.append("• Слишком частые тренировки могут вызывать переутомление. Добавьте день отдыха.")

    # --- Smoking ---
    if data.get("Smoking_status", 0) == 1:
        advice.append("• Курение ухудшает насыщение кислородом и мешает засыпанию. Попробуйте сократить количество сигарет.")

    # --- Sleep Duration ---
    sleep_dur = data.get("Sleep_duration", 7)
    if sleep_dur < 6:
        advice.append("• Попробуйте спать дольше 6 часов. Недосып снижает восстановление мозга и памяти.")
    elif sleep_dur > 9:
        advice.append("• Слишком долгий сон тоже может указывать на усталость или стресс. Оптимум — 7–9 часов.")

    # --- Wake and Bed Time ---
    wake = data.get("wake_hour", 7)
    bed = data.get("bed_hour", 23)
    # bed — это 12-часовое время, где 12 = полночь
    if bed in [12, 1, 2, 3, 4] or wake < 5:
        advice.append("• Вы ложитесь поздно или слишком рано встаёте. Попробуйте спать в диапазоне 22:00–7:00.")
    # --- Age ---
    age = data.get("Age", 25)
    if age > 60:
        advice.append("• С возрастом сон становится более поверхностным. Попробуйте вечерние расслабляющие практики (чай с ромашкой, медитация).")
    # --- Итог по качеству сна ---
    if label == 0:
        advice.append("\n😴 Ваш сон нуждается в улучшении. Попробуйте применить несколько советов выше.")
    elif label == 1:
        advice.append("\n🌙 У вас хороший сон! Продолжайте в том же духе.")
    elif label == 2:
        advice.append("\n🌀 Сон средний — можно улучшить, Попробуйте применить несколько советов выше.")

    return "\n".join(advice)

