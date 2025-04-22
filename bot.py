import logging
import os
import io
import qrcode
import numpy as np
import cv2
from PIL import Image, ImageOps
from telegram import Update, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Настройка логирования для отладки
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Функции для RGB QR ---

def create_rgb_qr(data_list):
    """Создает RGB QR-код из списка данных (до 3 строк)."""
    if not data_list or len(data_list) > 3:
        raise ValueError("Нужно от 1 до 3 строк данных, разделенных |")

    qr_images = []
    for data in data_list:
        if data: # Генерируем QR только если есть данные
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(data)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white").convert('L') # Ч/Б
            qr_images.append(np.array(img))
        else:
             qr_images.append(None) # Добавляем None если данных для канала нет


    if not any(img is not None for img in qr_images):
         raise ValueError("Хотя бы одна строка данных должна быть не пустой.")

    # Находим максимальный размер QR-кода
    max_size = 0
    valid_images = [img for img in qr_images if img is not None]
    if not valid_images:
         raise ValueError("Не удалось сгенерировать ни одного QR-кода.")

    base_img_np = valid_images[0]
    max_size = base_img_np.shape[0] # Берем размер первого валидного

    # Приводим все QR к одному размеру (если нужно) и создаем каналы
    channels = []
    default_channel = np.full((max_size, max_size), 255, dtype=np.uint8) # Белый фон по умолчанию

    for img_np in qr_images:
        if img_np is not None:
            # Если размер отличается, масштабируем (хотя qrcode обычно делает их одного размера)
            if img_np.shape[0] != max_size:
                 img_resized = cv2.resize(img_np, (max_size, max_size), interpolation=cv2.INTER_NEAREST)
                 channels.append(img_resized)
            else:
                 channels.append(img_np)
        else:
            # Если данных не было, добавляем пустой (белый) канал
            channels.append(default_channel)


    # Дополняем до 3 каналов, если нужно
    while len(channels) < 3:
         channels.append(default_channel)


    # Инвертируем цвета для каналов (черные модули QR станут 0 в канале)
    # OpenCV ожидает BGR, Pillow работает с RGB. Будем собирать как BGR для OpenCV, потом конвертируем если надо
    # Канал 0 -> Blue, 1 -> Green, 2 -> Red
    blue_channel = np.where(channels[0] == 0, 0, 255).astype(np.uint8)
    green_channel = np.where(channels[1] == 0, 0, 255).astype(np.uint8)
    red_channel = np.where(channels[2] == 0, 0, 255).astype(np.uint8)


    # Объединяем каналы в цветное изображение BGR
    colored_qr_np = cv2.merge([blue_channel, green_channel, red_channel])

    # Конвертируем из NumPy в PIL Image для отправки
    colored_qr_pil = Image.fromarray(colored_qr_np, 'RGB') # Создаем из BGR как RGB

    # Сохраняем в байтовый поток
    byte_stream = io.BytesIO()
    colored_qr_pil.save(byte_stream, format='PNG')
    byte_stream.seek(0)
    return byte_stream

def read_rgb_qr(image_bytes):
    """Читает данные из каналов RGB QR-кода."""
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
             return ["Ошибка: Не удалось декодировать изображение."]

        # Разделяем на каналы B, G, R
        blue_channel, green_channel, red_channel = cv2.split(img_bgr)

        channels = {'Blue': blue_channel, 'Green': green_channel, 'Red': red_channel}
        decoded_data = {}
        qr_detector = cv2.QRCodeDetector()

        for name, channel in channels.items():
            # Инвертируем канал для детектора (он ищет черные модули на белом фоне)
            # В нашем случае 0 в канале - это данные, 255 - фон.
            # Детектор может лучше работать с ч/б изображением
            channel_bw = np.where(channel == 0, 0, 255).astype(np.uint8)

            try:
                 data, _, _ = qr_detector.detectAndDecode(channel_bw)
                 if data:
                      decoded_data[name] = data
                 else:
                     # Попробуем инвертировать, если не нашлось
                     channel_inverted = np.where(channel == 0, 255, 0).astype(np.uint8)
                     data_inv, _, _ = qr_detector.detectAndDecode(channel_inverted)
                     if data_inv:
                         decoded_data[name] = data_inv
                     else:
                          decoded_data[name] = "[пусто или не читается]"

            except Exception as e:
                 logger.error(f"Ошибка при декодировании канала {name}: {e}")
                 decoded_data[name] = "[ошибка чтения]"

        return [f"{name}: {data}" for name, data in decoded_data.items()]

    except Exception as e:
         logger.error(f"Общая ошибка при чтении RGB QR: {e}")
         return ["Ошибка: Не удалось обработать изображение."]


# --- Команды и обработчики Телеграм ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Отправляет приветственное сообщение."""
    await update.message.reply_text(
        "Привет! Я бот для создания и чтения цветных RGB QR-кодов.\n"
        "Используй /create <текст1>|<текст2>|<текст3> для создания.\n"
        "Отправь мне фото RGB QR-кода для чтения.\n"
        "(Разделяй данные для каналов символом '|'. Можно указать 1, 2 или 3 блока данных)."
    )

async def create_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Создает RGB QR-код по команде."""
    if not context.args:
        await update.message.reply_text("Пожалуйста, укажи данные после команды /create.\n"
                                        "Пример: /create Привет|Мир|Бот")
        return

    combined_data = " ".join(context.args)
    data_list = [d.strip() for d in combined_data.split('|')]

    # Ограничим до 3х элементов
    if len(data_list) > 3:
        data_list = data_list[:3]
    while len(data_list) < 3: # Дополним пустыми строками для единообразия
        data_list.append("")


    try:
        logger.info(f"Создание RGB QR для данных: {data_list}")
        qr_image_stream = create_rgb_qr(data_list)
        await update.message.reply_photo(photo=InputFile(qr_image_stream, filename="rgb_qr.png"),
                                         caption=f"RGB QR-код создан для:\nB: {data_list[0]}\nG: {data_list[1]}\nR: {data_list[2]}")
        logger.info("RGB QR-код успешно отправлен.")
    except ValueError as ve:
         logger.warning(f"Ошибка значения при создании QR: {ve}")
         await update.message.reply_text(f"Ошибка: {ve}")
    except Exception as e:
        logger.error(f"Ошибка при создании RGB QR: {e}", exc_info=True)
        await update.message.reply_text("Произошла ошибка при создании QR-кода.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает полученное фото для чтения RGB QR."""
    if not update.message.photo:
        return

    logger.info("Получено фото для распознавания RGB QR.")
    # Берем фото наибольшего разрешения
    photo_file = await update.message.photo[-1].get_file()
    try:
        # Скачиваем фото в память
        photo_bytes = await photo_file.download_as_bytearray()
        logger.info(f"Фото скачано, размер: {len(photo_bytes)} байт.")

        # Декодируем
        decoded_results = read_rgb_qr(bytes(photo_bytes))
        logger.info(f"Результаты декодирования: {decoded_results}")

        await update.message.reply_text("Результаты чтения RGB QR:\n" + "\n".join(decoded_results))

    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}", exc_info=True)
        await update.message.reply_text("Не удалось обработать полученное фото.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Логирует ошибки, вызванные обновлениями."""
    logger.error("Exception while handling an update:", exc_info=context.error)


def main() -> None:
    """Запускает бота."""
    # Получаем токен из переменных окружения
    token = os.environ.get("BOT_TOKEN")
    if not token:
        logger.critical("Не найден BOT_TOKEN в переменных окружения!")
        return

    # Создаем приложение
    application = Application.builder().token(token).build()

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("create", create_command))
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo))

    # Добавляем обработчик ошибок
    application.add_error_handler(error_handler)

    # Запускаем бота
    logger.info("Запуск бота...")
    application.run_polling()

if __name__ == "__main__":
    main()
