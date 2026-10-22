import os
import re
import traceback
from datetime import datetime
from tkinter import Tk, filedialog, simpledialog, messagebox

import torch
import soundfile as sf
from PIL import Image, UnidentifiedImageError
from transformers import pipeline


IMAGE_MODEL = "Salesforce/blip-image-captioning-base"
VOICE_MODEL = "facebook/mms-tts-eng"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DESCRIPTIONS_FILE = os.path.join(BASE_DIR, "descriptions.txt")

SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png")

image_captioner = None
voice_generator = None

last_audio_file = None


def create_folders():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def choose_device():
    if torch.cuda.is_available():
        return 0
    return -1


def load_image_model():
    global image_captioner

    if image_captioner is None:
        print("\nЗагружается модель распознавания изображений...")
        image_captioner = pipeline(
            "image-to-text",
            model=IMAGE_MODEL,
            device=choose_device()
        )
        print("Модель распознавания загружена.")

    return image_captioner


def load_voice_model():
    global voice_generator

    if voice_generator is None:
        print("\nЗагружается модель озвучивания...")
        voice_generator = pipeline(
            "text-to-speech",
            model=VOICE_MODEL,
            device=choose_device()
        )
        print("Модель озвучивания загружена.")

    return voice_generator


def select_images():
    root = Tk()
    root.withdraw()

    paths = filedialog.askopenfilenames(
        title="Выберите изображения",
        initialdir=INPUT_DIR,
        filetypes=[
            ("Изображения", "*.jpg *.jpeg *.png"),
            ("Все файлы", "*.*")
        ]
    )

    root.destroy()
    return list(paths)


def validate_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError("Выбранный файл не найден.")

    extension = os.path.splitext(image_path)[1].lower()

    if extension not in SUPPORTED_FORMATS:
        raise ValueError(
            "Неподдерживаемый формат. Выберите JPG, JPEG или PNG."
        )

    try:
        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            image.convert("RGB").load()

    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(
            "Не удалось прочитать изображение. Возможно, файл повреждён."
        ) from error


def clean_filename(name):
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")
    return name or "audio"


def choose_audio_path(image_path):
    root = Tk()
    root.withdraw()

    default_name = os.path.splitext(
        os.path.basename(image_path)
    )[0] + "_description.wav"

    path = filedialog.asksaveasfilename(
        title="Сохранить аудио",
        initialdir=OUTPUT_DIR,
        initialfile=default_name,
        defaultextension=".wav",
        filetypes=[("WAV audio", "*.wav")]
    )

    root.destroy()

    if not path:
        return None

    if not path.lower().endswith(".wav"):
        path += ".wav"

    return path


def save_description(image_path, description, audio_path):
    with open(DESCRIPTIONS_FILE, "a", encoding="utf-8") as file:
        file.write("=" * 60 + "\n")
        file.write(f"Дата: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        file.write(f"Изображение: {os.path.basename(image_path)}\n")
        file.write(f"Описание: {description}\n")
        file.write(f"Модель изображения: {IMAGE_MODEL}\n")
        file.write(f"Модель озвучивания: {VOICE_MODEL}\n")
        file.write(f"Аудиофайл: {audio_path}\n\n")


def process_image(image_path, ask_audio_name=True):
    global last_audio_file

    try:
        validate_image(image_path)

        print("\n" + "=" * 60)
        print("Файл:", os.path.basename(image_path))
        print("Модель изображения:", IMAGE_MODEL)

        model = load_image_model()

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            result = model(image, max_new_tokens=100)

        description = result[0].get("generated_text", "").strip()

        if not description:
            print("Ошибка: модель не смогла создать описание.")
            return False

        print("\nОписание изображения:")
        print(description)

        if ask_audio_name:
            audio_path = choose_audio_path(image_path)
            if not audio_path:
                print("Сохранение аудио отменено.")
                return False
        else:
            name = clean_filename(
                os.path.splitext(os.path.basename(image_path))[0]
            )
            audio_path = os.path.join(
                OUTPUT_DIR, name + "_description.wav"
            )

        print("\nМодель озвучивания:", VOICE_MODEL)
        voice_model = load_voice_model()
        speech = voice_model(description)

        audio = speech["audio"]
        sampling_rate = speech["sampling_rate"]

        audio = audio.squeeze()

        sf.write(audio_path, audio, sampling_rate)

        last_audio_file = audio_path

        print("\nАудиофайл сохранён:")
        print(audio_path)

        save_description(image_path, description, audio_path)

        print("Описание добавлено в descriptions.txt")
        print("Анализ завершён успешно.")

        return True

    except Exception as error:
        print("\nНе удалось обработать изображение.")
        print("Причина:", error)
        print(
            "Проверьте подключение к интернету, формат изображения "
            "и установленные библиотеки."
        )
        return False


def process_multiple_images():
    paths = select_images()

    if not paths:
        print("Изображения не выбраны.")
        return

    successful = 0

    for path in paths:
        if process_image(path, ask_audio_name=False):
            successful += 1

    print("\nОбработка завершена.")
    print(f"Успешно обработано: {successful} из {len(paths)}")


def process_input_folder():
    paths = []

    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(SUPPORTED_FORMATS):
            paths.append(os.path.join(INPUT_DIR, filename))

    if not paths:
        print("В папке input нет изображений JPG, JPEG или PNG.")
        return

    successful = 0

    for path in paths:
        if process_image(path, ask_audio_name=False):
            successful += 1

    print(f"\nУспешно обработано: {successful} из {len(paths)}")


def replay_audio():
    root = Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Выберите аудиофайл для прослушивания",
        initialdir=OUTPUT_DIR,
        filetypes=[("WAV audio", "*.wav")]
    )

    root.destroy()

    if not path:
        print("Аудиофайл не выбран.")
        return

    if not os.path.isfile(path):
        print("Аудиофайл не найден.")
        return

    try:
        if os.name == "nt":
            os.startfile(path)
        else:
            import subprocess
            import sys

            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])

        print("Открыт аудиофайл:", path)

    except Exception as error:
        print("Не удалось открыть аудио:", error)


def show_models():
    print("\nМОДЕЛИ ПРОЕКТА")
    print("-" * 50)
    print("1. Описание изображений:")
    print(IMAGE_MODEL)
    print("https://huggingface.co/" + IMAGE_MODEL)
    print("\n2. Синтез английской речи:")
    print(VOICE_MODEL)
    print("https://huggingface.co/" + VOICE_MODEL)


def main():
    create_folders()

    print("=" * 50)
    print("AI VISION & VOICE ASSISTANT")
    print("Распознавание изображений и голосовой помощник")
    print("=" * 50)

    while True:
        print("\nМЕНЮ")
        print("1. Выбрать и обработать изображение")
        print("2. Обработать несколько изображений")
        print("3. Обработать все изображения из папки input")
        print("4. Повторно прослушать аудио")
        print("5. Информация о моделях")
        print("0. Выход")

        choice = input("\nВыберите действие: ").strip()

        if choice == "1":
            paths = select_images()

            if paths:
                process_image(paths[0], ask_audio_name=True)
            else:
                print("Изображение не выбрано.")

        elif choice == "2":
            process_multiple_images()

        elif choice == "3":
            process_input_folder()

        elif choice == "4":
            replay_audio()

        elif choice == "5":
            show_models()

        elif choice == "0":
            print("Программа завершена.")
            break

        else:
            print("Неверный пункт меню. Попробуйте ещё раз.")


if __name__ == "__main__":
    main()
