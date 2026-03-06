import os
import re
import json
import math
import asyncio
import subprocess
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from openai import OpenAI

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import FSInputFile


# =====================================================
# ENV
# =====================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set in .env")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env")

bot = Bot(BOT_TOKEN)
dp = Dispatcher()
client = OpenAI(api_key=OPENAI_API_KEY)


# =====================================================
# FILES / STORAGE
# =====================================================

USERS_FILE = "users.json"
PROJECTS_FILE = "projects.json"
HISTORY_FILE = "history.json"
QUESTIONS_FILE = "questions.json"
SESSIONS_FILE = "sessions.json"
TRANSCRIBE_MODE_FILE = "transcribe_mode.json"

# "встреча" завершается, если N секунд нет новых аудио
MEETING_TIMEOUT_SECONDS = 45

# лимит Telegram на сообщение (безопаснее 3500)
TG_TEXT_CHUNK = 3500

# лимит модели транскрибации по длительности:
# в твоих логах было: max 1400 sec, поэтому режем по 1200 с запасом
TRANSCRIBE_MAX_SECONDS = 1200

# папка для временных файлов (чтобы не мусорить в корне)
TMP_DIR = "tmp"
os.makedirs(TMP_DIR, exist_ok=True)


# =====================================================
# UI (KEYBOARD)
# =====================================================

main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🧠 Состояние")],
        [KeyboardButton(text="🎤 Отчёт встречи")],
        [KeyboardButton(text="📝 Расшифровка")],
        [KeyboardButton(text="👤 Моя роль")],
    ],
    resize_keyboard=True,
)


# =====================================================
# PROMPTS
# =====================================================

SYSTEM_PROMPT = """
Ты — операционный помощник директора архитектурного бюро.

ВАЖНО:
Компания НЕ строит дома.
Основной продукт компании — АРХИТЕКТУРНЫЙ ПРОЕКТ и проектная документация.
Дом — это результат работы клиента со строителями.
Результат работы компании — проект/документация/согласования/авторский надзор (опционально).

Типичный путь проектной работы:
1) Первичное обращение клиента
2) Выявление потребностей
3) Коммерческое предложение
4) Концепция / эскиз
5) Архитектурный проект (основной результат)
6) Рабочая документация
7) Авторский надзор (опционально)

Риски:
- клиент не определился с бюджетом
- размытые требования
- бесконечные изменения
- ожидания не согласованы
- затягивание согласований
- неоплаченные этапы

Правила:
- Если данных мало — задай максимум 3–5 уточняющих вопросов.
- После 2 раундов уточнений ОБЯЗАТЕЛЬНО переходи к отчёту.
- Не уходи в технические детали проектирования (марки арматуры, узлы и т.п.). Ты — про управление и контроль.
- Статус проекта трактуем так:
  🟢 GREEN — всё под контролем (ключевые решения/рамки понятны)
  🟡 YELLOW — есть нерешённые вопросы/риски (требует решений)
  🔴 RED — зона риска/срыв/конфликт/неоплата/критические блокеры

Формат ответа:

🧠 Помощник директора

✅ Решения
⚠️ Упущено
📋 Задачи

Состояние проекта: GREEN / YELLOW / RED
Причина — управленческая, 1 предложение
👉 Следующий шаг — 1 чёткий шаг

В конце обязательно:

MEMORY_UPDATE:
{
  "project_name":"название проекта или клиент",
  "health":"YELLOW",
  "next_step":"..."
}
""".strip()

SUMMARY_PROMPT = """
Сделай короткое summary расшифровки встречи (2–6 буллетов):
- главная цель/контекст
- ключевые решения
- риски/неясности
- ближайшие шаги

Без воды. На русском.
""".strip()


# =====================================================
# JSON HELPERS
# =====================================================

def load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_users() -> Dict[str, str]:
    return load_json(USERS_FILE, {})


def save_users(data: Dict[str, str]) -> None:
    save_json(USERS_FILE, data)


def load_projects() -> Dict[str, Dict[str, Any]]:
    return load_json(PROJECTS_FILE, {})


def save_projects(data: Dict[str, Dict[str, Any]]) -> None:
    save_json(PROJECTS_FILE, data)


def load_history() -> Dict[str, List[Dict[str, str]]]:
    return load_json(HISTORY_FILE, {})


def save_history(data: Dict[str, List[Dict[str, str]]]) -> None:
    save_json(HISTORY_FILE, data)


def add_to_history(chat_id: str, text: str, keep_last: int = 6) -> None:
    """
    История нужна, чтобы модель меньше "забывала" контекст диалога.
    Храним только последние N сообщений, чтобы не раздувать запрос.
    """
    history = load_history()
    h = history.get(chat_id, [])
    h.append({"role": "user", "content": text})
    history[chat_id] = h[-keep_last:]
    save_history(history)


def get_history(chat_id: str) -> List[Dict[str, str]]:
    return load_history().get(chat_id, [])


def load_questions() -> Dict[str, int]:
    return load_json(QUESTIONS_FILE, {})


def save_questions(data: Dict[str, int]) -> None:
    save_json(QUESTIONS_FILE, data)


def load_sessions() -> Dict[str, Dict[str, Any]]:
    return load_json(SESSIONS_FILE, {})


def save_sessions(data: Dict[str, Dict[str, Any]]) -> None:
    save_json(SESSIONS_FILE, data)


def load_transcribe_mode() -> Dict[str, bool]:
    return load_json(TRANSCRIBE_MODE_FILE, {})


def save_transcribe_mode(data: Dict[str, bool]) -> None:
    save_json(TRANSCRIBE_MODE_FILE, data)


# =====================================================
# PROJECT MATCHING
# =====================================================

def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def find_or_create_project(projects: Dict[str, Any], name: str) -> str:
    """
    Авто-определение: если имя совпадает (после нормализации) — считаем это тем же проектом.
    Иначе создаём новый.
    """
    norm = normalize_name(name)
    for existing in list(projects.keys()):
        if normalize_name(existing) == norm:
            return existing

    projects[name] = projects.get(name, {})
    return name


# =====================================================
# TELEGRAM TEXT HELPERS
# =====================================================

async def send_long_message(chat_id: str, text: str) -> None:
    """
    Отправка длинного текста частями (Telegram ограничивает длину сообщения).
    """
    if not text:
        return

    parts = [text[i:i + TG_TEXT_CHUNK] for i in range(0, len(text), TG_TEXT_CHUNK)]
    for p in parts:
        await bot.send_message(chat_id, p)


def status_humanize(text: str) -> str:
    """
    Меняем слова GREEN/YELLOW/RED на понятные статусы с иконками.
    """
    return (
        text.replace("GREEN", "🟢 Всё под контролем")
            .replace("YELLOW", "🟡 Требует решений")
            .replace("RED", "🔴 Зона риска")
    )


# =====================================================
# FILE OUTPUT (TRANSCRIPT + SUMMARY)
# =====================================================

async def send_transcript_file_with_summary(chat_id: str, transcript: str) -> None:
    """
    Твоё требование:
    - длинное — не в чат, а текстовым файлом
    - в сообщении — короткое summary
    """
    transcript = transcript.strip()
    if not transcript:
        await bot.send_message(chat_id, "Похоже, в записи не удалось распознать речь.")
        return

    def _summary_call() -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": transcript[:12000]},
            ],
        )

        content = resp.choices[0].message.content
        return (content or "").strip()

    try:
        summary = await asyncio.to_thread(_summary_call)
    except Exception:
        summary = ""

    if not summary:
        summary = "• (summary не удалось получить, см. файл расшифровки)"

    # 2) пишем .txt файл
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(TMP_DIR, f"transcript_{chat_id}_{ts}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(transcript)

    # 3) отправляем документ корректно (FSInputFile)
    doc = FSInputFile(file_path)

    caption = f"🧾 Summary:\n{summary}"
    # Telegram caption тоже ограничен, подрежем
    if len(caption) > 900:
        caption = caption[:900] + "…"
        
    await bot.send_document(chat_id=chat_id, document=doc, caption=caption)

    # 4) чистим файл
    try:
        os.remove(file_path)
    except Exception:
        pass


# =====================================================
# FFMPEG AUDIO UTILS (NO PYDUB)
# =====================================================

def ffprobe_duration_seconds(input_path: str) -> float:
    """
    Достаём длительность через ffprobe.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            input_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    raw = (result.stdout or "").strip()
    if not raw:
        raise RuntimeError(f"ffprobe не смог прочитать длительность: {result.stderr}")
    return float(raw)


def split_long_audio_to_mp3_chunks(input_path: str, chat_id: str) -> List[str]:
    """
    Режем любой входной формат на mp3 чанки, пригодные для транскрибации.
    """
    duration = ffprobe_duration_seconds(input_path)

    if duration <= TRANSCRIBE_MAX_SECONDS:
        # всё равно перегоняем в mp3 (чтобы не ловить 'Unsupported file format')
        out = os.path.join(TMP_DIR, f"chunk_{chat_id}_0.mp3")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ac", "1",
                "-ar", "16000",
                "-b:a", "64k",
                out,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return [out]

    parts = math.ceil(duration / TRANSCRIBE_MAX_SECONDS)
    chunks: List[str] = []

    for i in range(parts):
        start = i * TRANSCRIBE_MAX_SECONDS
        out = os.path.join(TMP_DIR, f"chunk_{chat_id}_{i}.mp3")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-ss", str(start),
                "-t", str(TRANSCRIBE_MAX_SECONDS),
                "-ac", "1",
                "-ar", "16000",
                "-b:a", "64k",
                out,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        chunks.append(out)

    return chunks


# =====================================================
# DOWNLOAD HELPERS
# =====================================================

def is_google_drive_link(text: str) -> bool:
    return "drive.google.com" in (text or "")


def extract_drive_file_id(url: str) -> Optional[str]:
    # варианты ссылок:
    # https://drive.google.com/file/d/<ID>/view?...
    # https://drive.google.com/open?id=<ID>
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    m = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        return m.group(1)
    return None


def download_from_google_drive(url: str, output_path: str) -> None:
    """
    Скачиваем файл с Google Drive.
    Важно: это синхронная функция — зови её через asyncio.to_thread.
    """
    file_id = extract_drive_file_id(url)
    if not file_id:
        raise ValueError("Не смог извлечь FILE_ID из ссылки Google Drive")

    session = requests.Session()
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    r = session.get(download_url, stream=True, timeout=300)
    r.raise_for_status()

    # подтверждение для больших файлов
    for k, v in r.cookies.items():
        if k.startswith("download_warning"):
            download_url = f"{download_url}&confirm={v}"
            r = session.get(download_url, stream=True, timeout=300)
            r.raise_for_status()
            break

    with open(output_path, "wb") as f:
        for chunk in r.iter_content(1024 * 64):
            if chunk:
                f.write(chunk)


def download_telegram_file_bytes(file_url: str) -> bytes:
    """
    Синхронно скачиваем bytes по URL. В async используем asyncio.to_thread.
    """
    r = requests.get(file_url, timeout=120)
    r.raise_for_status()
    return r.content


# =====================================================
# OPENAI TRANSCRIBE / ANALYZE
# =====================================================

def transcribe_file(path: str) -> str:
    """
    Синхронная транскрибация одного чанка.
    """
    with open(path, "rb") as f:
        t = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
        ).text
    return (t or "").strip()


async def transcribe_any_audio_to_text(input_path: str, chat_id: str) -> str:
    """
    Главный пайплайн:
    1) режем на mp3 чанки (ffmpeg)
    2) транскрибируем чанки
    3) склеиваем
    """
    # нарезка через ffmpeg — тяжёлая операция, но быстрая
    chunks = await asyncio.to_thread(split_long_audio_to_mp3_chunks, input_path, chat_id)

    full_parts: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        # редкое обновление прогресса (не спамим)
        await bot.send_message(chat_id, f"🎧 Транскрибация: часть {idx}/{len(chunks)}")

        try:
            part = await asyncio.to_thread(transcribe_file, chunk)
        except Exception as e:
            raise RuntimeError(f"Ошибка транскрибации части {idx}: {e}")

        if part:
            full_parts.append(part)

    # чистим чанки
    for c in chunks:
        try:
            os.remove(c)
        except Exception:
            pass

    return "\n\n".join(full_parts).strip()


async def analyze_meeting_text(chat_id: str, meeting_text: str) -> str:
    """
    Генерируем отчёт "помощник директора" с лимитом уточнений.
    """
    q = load_questions()
    count = int(q.get(chat_id, 0))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Уточнений уже было: {count}. После 2 переходи к отчёту."},
        *get_history(chat_id),
        {"role": "user", "content": meeting_text},
    ]

    def _call():
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return (resp.choices[0].message.content or "").strip()

    result = await asyncio.to_thread(_call)

    # счётчик уточнений
    if result.startswith("Нужно уточнить"):
        q[chat_id] = count + 1
    else:
        q[chat_id] = 0
    save_questions(q)

    return result


def parse_memory_update(result_text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """
    Возвращаем (visible_text, memory_json_or_none)
    """
    if "MEMORY_UPDATE:" not in result_text:
        return result_text, None

    try:
        visible, mem_raw = result_text.split("MEMORY_UPDATE:", 1)
        mem = json.loads(mem_raw.strip())
        return visible.strip(), mem
    except Exception:
        # если JSON кривой — лучше не ломать ответ
        return result_text, None


def save_project_memory(mem: Dict[str, Any]) -> None:
    projects = load_projects()
    project_name = str(mem.get("project_name") or "Без названия").strip()
    key = find_or_create_project(projects, project_name)

    health = str(mem.get("health") or "YELLOW").upper()
    if health not in ("GREEN", "YELLOW", "RED"):
        health = "YELLOW"

    projects[key] = {
        "health": health,
        "next_step": str(mem.get("next_step") or "").strip(),
        "last_contact": str(date.today()),
    }
    save_projects(projects)


# =====================================================
# MEETING SESSION (COLLECT AUDIO UNTIL TIMEOUT)
# =====================================================

async def finalize_meeting(chat_id: str) -> None:
    """
    Финализация встречи: собираем тексты из sessions.json и делаем один отчёт.
    """
    sessions = load_sessions()
    session = sessions.get(chat_id)
    if not session:
        return

    texts = session.get("messages", [])
    if not texts:
        return

    # очищаем сессию сразу, чтобы не дублить финализацию
    sessions.pop(chat_id, None)
    save_sessions(sessions)

    full_text = "\n\n".join(texts).strip()
    if not full_text:
        await bot.send_message(chat_id, "Похоже, встреча пустая — нечего анализировать.")
        return

    # добавим в историю (чтобы модель держала контекст проекта)
    add_to_history(chat_id, full_text)

    result = await analyze_meeting_text(chat_id, full_text)

    visible, mem = parse_memory_update(result)
    if mem:
        try:
            save_project_memory(mem)
        except Exception:
            pass

    visible = status_humanize(visible).strip()
    await send_long_message(chat_id, visible)


async def meeting_timer(chat_id: str) -> None:
    """
    Таймер на завершение встречи. НЕ блокирует обработку новых сообщений.
    """
    await asyncio.sleep(MEETING_TIMEOUT_SECONDS)

    sessions = load_sessions()
    session = sessions.get(chat_id)
    if not session:
        return

    last_time = float(session.get("last_time") or 0)
    if datetime.now().timestamp() - last_time >= MEETING_TIMEOUT_SECONDS:
        await finalize_meeting(chat_id)


def touch_session(chat_id: str, transcript_piece: str) -> None:
    """
    Добавляем кусок расшифровки в текущую сессию встречи.
    """
    sessions = load_sessions()
    session = sessions.get(chat_id, {"messages": [], "last_time": 0, "timer_started": False})
    

    session["messages"].append(transcript_piece)
    session["last_time"] = datetime.now().timestamp()

    sessions[chat_id] = session
    save_sessions(sessions)


# =====================================================
# COMMANDS / BUTTONS
# =====================================================

@dp.message(CommandStart())
async def start(message: Message) -> None:
    users = load_users()
    users[str(message.chat.id)] = users.get(str(message.chat.id), "director")
    save_users(users)

    await message.answer("🧠 Помощник директора активирован.", reply_markup=main_keyboard)


@dp.message(lambda m: m.text == "🎤 Отчёт встречи")
async def meeting_hint(message: Message) -> None:
    await message.answer(
        "Отправь голосовые/аудио после встречи.\n"
        "Я соберу всё в одну встречу и через ~45 секунд тишины пришлю итоговый отчёт.\n\n"
        "Если запись слишком длинная и Telegram ругается — закинь в Google Drive и пришли ссылку."
    )


@dp.message(lambda m: m.text == "📝 Расшифровка")
async def enable_transcribe_mode(message: Message) -> None:
    modes = load_transcribe_mode()
    modes[str(message.chat.id)] = True
    save_transcribe_mode(modes)

    await message.answer(
        "Ок. Следующее аудио/голосовое/ссылка будет обработано как *просто расшифровка*.\n"
        "Я пришлю .txt файлом + короткое summary."
    )


@dp.message(lambda m: m.text == "👤 Моя роль")
async def role_button(message: Message) -> None:
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="👔 Директор", callback_data="role_director")],
        [InlineKeyboardButton(text="📐 Архитектор", callback_data="role_architect")],
        [InlineKeyboardButton(text="💰 Продажи", callback_data="role_sales")],
    ])
    await message.answer("Выбери роль:", reply_markup=keyboard)


@dp.callback_query(lambda c: c.data and c.data.startswith("role_"))
async def role_callback(callback) -> None:
    role = callback.data.replace("role_", "")

    users = load_users()
    users[str(callback.message.chat.id)] = role
    save_users(users)

    await callback.message.answer(f"✅ Роль установлена: {role}")
    await callback.answer()


@dp.message(lambda m: m.text == "🧠 Состояние")
async def brain(message: Message) -> None:
    projects = load_projects()
    if not projects:
        await message.answer("Проектов пока нет.")
        return

    green = yellow = red = 0
    lines: List[str] = []

    for name, info in projects.items():
        h = str(info.get("health") or "").upper()
        if h == "GREEN":
            green += 1
        elif h == "YELLOW":
            yellow += 1
            lines.append(f"🟡 {name}")
        elif h == "RED":
            red += 1
            lines.append(f"🔴 {name}")

    text = (
        "🧠 Операционная сводка\n\n"
        f"🟢 Под контролем: {green}\n"
        f"🟡 Требуют решений: {yellow}\n"
        f"🔴 Зона риска: {red}\n\n"
    )

    if lines:
        text += "⚠️ Фокус:\n" + "\n".join(lines)
    else:
        text += "✅ Критических зон сейчас нет."

    await message.answer(text)


# =====================================================
# AUDIO HANDLER (TELEGRAM)
# =====================================================

@dp.message(lambda m: m.voice or m.audio)
async def handle_audio(message: Message) -> None:
    """
    Главный вход для голосовых/аудио в Telegram.
    Важно:
    - не спамим "слушаю/анализирую"
    - тяжёлую работу делаем в фоне (create_task)
    """
    await message.answer("🎧 Запись получена. Обрабатываю…")
    asyncio.create_task(process_audio_message(message))


async def process_audio_message(message: Message) -> None:
    chat_id = str(message.chat.id)

    # режим "только расшифровка"
    modes = load_transcribe_mode()
    transcribe_mode = bool(modes.get(chat_id, False))

    input_path = ""
    try:
        # 1) получаем file_id
        file_id = message.voice.file_id if message.voice else message.audio.file_id

        # 2) пытаемся получить file_path (может упасть "file is too big")
        try:
            file = await bot.get_file(file_id)
        except TelegramBadRequest as e:
            # это тот самый кейс: Telegram server says - Bad Request: file is too big
            await bot.send_message(
                chat_id,
                "❌ Telegram не даёт скачать этот файл (слишком большой для Bot API).\n"
                "Решение: загрузи аудио в Google Drive и пришли ссылку сюда — я обработаю."
            )
            return

        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file.file_path}"

        # 3) скачиваем байты синхронно в отдельном потоке
        data = await asyncio.to_thread(download_telegram_file_bytes, file_url)

        # 4) сохраняем с нормальным расширением (не критично, ffmpeg прочитает)
        ext = "ogg" if message.voice else "bin"
        if message.audio and message.audio.file_name and "." in message.audio.file_name:
            ext = message.audio.file_name.rsplit(".", 1)[-1].lower()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = os.path.join(TMP_DIR, f"tg_audio_{chat_id}_{ts}.{ext}")

        with open(input_path, "wb") as f:
            f.write(data)

        # 5) транскрибируем (с нарезкой и конвертацией)
        transcript = await transcribe_any_audio_to_text(input_path, chat_id)

    except Exception as e:
        await bot.send_message(chat_id, f"❌ Ошибка обработки аудио:\n{e}")
        return
    finally:
        if input_path:
            try:
                os.remove(input_path)
            except Exception:
                pass

    # --- дальше логика режима
    if transcribe_mode:
        # сбрасываем режим после одного использования
        modes[chat_id] = False
        save_transcribe_mode(modes)

        await send_transcript_file_with_summary(chat_id, transcript)
        return

    # иначе — режим встречи
    touch_session(chat_id, transcript)
    asyncio.create_task(meeting_timer(chat_id))


# =====================================================
# GOOGLE DRIVE LINK HANDLER
# =====================================================

@dp.message(lambda m: m.text and is_google_drive_link(m.text))
async def handle_drive_link(message: Message) -> None:
    """
    Если Telegram не тянет большие файлы — даём вариант: Google Drive ссылкой.
    """
    chat_id = str(message.chat.id)

    await message.answer("🔗 Ссылка получена. Скачиваю и обрабатываю…")

    # режим "только расшифровка"
    modes = load_transcribe_mode()
    transcribe_mode = bool(modes.get(chat_id, False))

    url = (message.text or "").strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_path = os.path.join(TMP_DIR, f"drive_audio_{chat_id}_{ts}.m4a")

    try:
        # скачивание drive — синхронное, кидаем в отдельный поток
        await asyncio.to_thread(download_from_google_drive, url, input_path)

        transcript = await transcribe_any_audio_to_text(input_path, chat_id)

    except Exception as e:
        await message.answer(f"❌ Ошибка скачивания/обработки Google Drive:\n{e}")
        return
    finally:
        try:
            os.remove(input_path)
        except Exception:
            pass

    if transcribe_mode:
        modes[chat_id] = False
        save_transcribe_mode(modes)

        await send_transcript_file_with_summary(chat_id, transcript)
        return

    touch_session(chat_id, transcript)
    asyncio.create_task(meeting_timer(chat_id))


# =====================================================
# DAILY CHECK (optional)
# =====================================================

async def daily_check() -> None:
    while True:
        await asyncio.sleep(86400)
        users = load_users()
        projects = load_projects()

        for chat_id in users.keys():
            for name, info in projects.items():
                if str(info.get("health") or "").upper() == "RED":
                    await bot.send_message(chat_id, f"🚨 Проект «{name}» в зоне риска.")


# =====================================================
# RUN
# =====================================================

async def main() -> None:
    asyncio.create_task(daily_check())
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
