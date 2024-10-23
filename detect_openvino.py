# import os
# import cv2
# import numpy as np
# from PIL import Image
# from utils.datasets import letterbox  # Используем функцию из YOLOv7 для изменения размера
#
#
# # 1. Функция для извлечения кадров из видео
# def extract_frames(video_path: str, output_folder: str, frame_rate: int = 1):
#
#     # Открываем видео
#     cap = cv2.VideoCapture(video_path)
#
#     # Создаем папку для сохранения кадров, если ее не существует
#     os.makedirs(output_folder, exist_ok=True)
#
#     frame_count = 0  # Счетчик кадров
#
#     while True:
#         # Читаем кадр из видео
#         success, frame = cap.read()
#
#         # Прекращаем цикл, если кадры закончились
#         if not success:
#             break
#
#         # Сохраняем кадр каждые N-кадров, где N = frame_rate
#         if frame_count % frame_rate == 0:
#             frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
#             cv2.imwrite(frame_filename, frame)  # Сохраняем кадр в формате JPG
#
#         frame_count += 1  # Увеличиваем счетчик кадров
#
#     # Освобождаем видеозапись
#     cap.release()
#
#
# # 2. Функция для предобработки изображения
# def preprocess_image(img_path: str):
#
#     # Открытие изображения с помощью PIL и преобразование в формат numpy
#     img0 = np.array(Image.open(img_path))  # Оригинальное изображение
#
#     # Изменение размера изображения до 640x640 с сохранением пропорций
#     img = letterbox(img0, new_shape=640, auto=False)[0]
#
#     # Преобразование изображения в формат (C, H, W), RGB
#     img = img.transpose(2, 0, 1)  # Преобразование из HWC в CHW (формат для модели)
#     img = np.ascontiguousarray(img)  # Преобразуем в непрерывный массив
#
#     return img, img0  # Возвращаем предобработанное и оригинальное изображение
#
#
# # 3. Функция для преобразования изображения в формат тензора для YOLOv7
# def prepare_input_tensor(image: np.ndarray):
#
#     input_tensor = image.astype(np.float32)  # Преобразуем в float32
#     input_tensor /= 255.0  # Нормализуем от 0 до 1
#
#     # Добавляем размерность для батча, если у изображения только 3 измерения
#     if input_tensor.ndim == 3:
#         input_tensor = np.expand_dims(input_tensor, 0)
#
#     return input_tensor
#
#
# # 4. Пример использования всего кода:
#
# # Извлечение кадров из видео
# video_path = 'video/vid.mp4'  # Укажи путь к твоему видеофайлу
# output_folder = 'extracted_frames'  # Папка, в которую будут сохраняться кадры
# extract_frames(video_path, output_folder, frame_rate=30)  # Извлекаем каждый 30-й кадр
#
# # Предобработка одного из извлечённых кадров
# img_path = 'extracted_frames/frame_0'


# import os
# import numpy as np
# import torch
#
# from PIL import Image
# from openvino.runtime import Core
# from utils.datasets import letterbox
# from utils.plots import plot_one_box
# from utils.general import non_max_suppression, scale_coords
#
# # 1. Определение имен классов и цветов для визуализации
# NAMES = ["cattle", "fallen_tree", "human", "power_line", "tractor"]  # Укажи свои классы
#
# # Цвета для визуализации
# COLORS = {name: [np.random.randint(0, 255) for _ in range(3)]
#           for i, name in enumerate(NAMES)}
#
# # 2. Функция для предобработки изображения
# def preprocess_image(img_path: str):
#     img0 = np.array(Image.open(img_path))  # Оригинальное изображение
#     img = letterbox(img0, new_shape=640, auto=False)[0]  # Изменение размера
#     img = img.transpose(2, 0, 1)  # BGR to RGB
#     img = np.ascontiguousarray(img)
#     return img, img0  # Возвращаем предобработанное и оригинальное изображение
#
# # 3. Функция для преобразования изображения в тензор
# def prepare_input_tensor(image: np.ndarray):
#     input_tensor = image.astype(np.float32)  # Преобразуем в float32
#     input_tensor /= 255.0  # Нормализация: 0-255 в 0.0-1.0
#     if input_tensor.ndim == 3:
#         input_tensor = np.expand_dims(input_tensor, 0)  # Добавление размерности для батча
#     return input_tensor
#
# # 4. Загрузка модели
# def load_model(model_path: str):
#     core = Core()
#     model = core.read_model(model_path)
#     compiled_model = core.compile_model(model, 'CPU')
#     return compiled_model
#
# # 5. Инференс модели на изображении
# def detect(model, image):
#     input_tensor = prepare_input_tensor(image)  # Подготовка тензора
#     predictions = model(input_tensor)  # Получаем предсказания
#
#     # Отладочные выводы
#     print(f"Predictions: {predictions}")  # Печатаем все предсказания
#     print(f"Output keys: {predictions.keys()}")  # Печатаем ключи предсказаний
#
#     output_key = next(iter(predictions.keys()))  # Получаем первый ключ
#     predictions_array = predictions[output_key]  # Убираем .numpy() здесь
#
#     # Печать формы выходных данных для отладки
#     print(f"Predictions shape: {predictions_array.shape}")  # Выводим форму предсказаний
#
#     # Проверка размерности
#     if len(predictions_array.shape) < 3:
#         print("Predictions do not have the expected shape. Exiting.")
#         return None
#
#     # Преобразуем массив NumPy в тензор PyTorch
#     predictions_tensor = torch.from_numpy(predictions_array)
#
#     # Выполняем non_max_suppression на тензоре
#     detections = non_max_suppression(predictions_tensor, conf_thres=0.25, iou_thres=0.45)
#     return detections
#
#
# # 6. Постобработка и визуализация
# def visualize_detections(image, detections, names, colors, input_tensor_shape):
#     if detections is not None and detections.size(0) > 0:  # Проверка на наличие детекций
#         for det in detections:
#             if det is not None and det.dim() == 2 and det.size(0) > 0:  # Проверка на размерность
#                 det[:, :4] = scale_coords(input_tensor_shape[2:], det[:, :4], image.shape).round()
#                 for *xyxy, conf, cls in det:
#                     label = f'{names[int(cls)]} {conf:.2f}'
#                     plot_one_box(xyxy, image, label=label, color=colors[names[int(cls)]], line_thickness=1)
#             else:
#                 print("Empty or invalid detection tensor.")
#     else:
#         print("No detections found.")
#     return image
#
#
# # Пример использования
# input_folder = 'extracted_frames/'  # Папка с кадрами
# model_path = 'model1/yolov7_fp32.xml'  # Укажи путь к твоей модели
#
# # 1. Загрузка модели
# compiled_model = load_model(model_path)
#
# # 2. Обработка извлечённых кадров
# for frame in os.listdir(input_folder):
#     img_path = os.path.join(input_folder, frame)
#     preprocessed_img, orig_img = preprocess_image(img_path)  # Предобработка изображения
#     detections = detect(compiled_model, preprocessed_img)  # Инференс
#
#     # Вывод для отладки
#     print(f"Detections for {frame}: {detections}")
#
#     # Инициализация переменной
#     image_with_boxes = orig_img.copy()  # Инициализация оригинальным изображением
#
#     # Проверка на наличие детекций
#     if detections and len(detections) > 0:
#         if isinstance(detections[0], torch.Tensor):
#             if detections[0].dim() == 2 and detections[0].size(0) > 0:  # Проверка на размерность
#                 image_with_boxes = visualize_detections(orig_img, detections[0], NAMES, COLORS,
#                                                         preprocessed_img.shape)  # Визуализация
#             else:
#                 print(
#                     f"No valid detections for {frame}. Shape: {detections[0].shape}")  # Сообщение, если нет валидных детекций
#         else:
#             print(f"Unexpected format for detections: {type(detections[0])}")
#     else:
#         print(f"No detections for {frame}.")  # Сообщение, если нет детекций
#
#     # Сохранение результата
#     result_img_path = os.path.join(input_folder, f"detected_{frame}")
#     Image.fromarray(image_with_boxes).save(result_img_path)
#
# print("Детекция завершена!")












