import cv2
import os
from datetime import datetime

# Укажите путь к видеофайлу
video_path = "../datasets/videos/50"

# Укажите путь и название папки для сохранения кадров
folder_path = '../datasets/source_images3'
folder_name = 'stand'

# Открываем видеофайл
video = cv2.VideoCapture(video_path)

# Определяем количество фреймов в видео
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Создаем папку для сохранения кадров
folder_path_with_name = os.path.join(folder_path, f'{folder_name}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{frame_count}-shorts')
os.makedirs(folder_path_with_name, exist_ok=True)

# Считываем фреймы и сохраняем их в папку
for i in range(frame_count):
    ret, frame = video.read()
    if not ret:
        break
    filename = os.path.join(folder_path_with_name, f'{i+1:05d}.jpg')
    cv2.imwrite(filename, frame)

# Освобождаем ресурсы
video.release()
cv2.destroyAllWindows()
