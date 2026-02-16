import os
from PIL import Image


def crop_images(input_dir, output_dir, center_x, center_y, width, height):
    # Создаем выходную директорию, если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Проходим по всем файлам в директории
    for filename in os.listdir(input_dir):
        # Проверяем, что файл - изображение (по расширению)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')):
            # Открываем изображение
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path)

            # Вычисляем координаты для обрезки (центрированная обрезка)
            left = (center_x - width)
            top = (center_y - height)
            right = (center_x + width)
            bottom = (center_y + height)

            # Обрезаем изображение
            cropped_img = img.crop((left, top, right, bottom))

            # Сохраняем в выходную директорию с тем же именем и форматом
            output_path = os.path.join(output_dir, filename)
            cropped_img.save(output_path)

            print(f"Обработано: {filename}")


# Пример использования
crop_images("C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\13.02.26", "C:\\Users\\artur\\Desktop\\13.02.26 sbs new\\fb_crop", 1075, 929, 150, 150)