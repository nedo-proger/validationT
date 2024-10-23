# import os
# import torch
# from torchvision import transforms
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
#
# class CustomDataset(Dataset):
#     def __init__(self, data_directory, transform=None):
#         self.data_directory = data_directory
#         self.transform = transform
#         # Список файлов изображений (например, .jpg или .png)
#         self.image_files = [f for f in os.listdir(data_directory) if f.endswith('.jpg') or f.endswith('.png')]
#         self.annotations = self.load_annotations()
#
#     def load_annotations(self):
#         # Реализуйте логику загрузки аннотаций здесь
#         annotations = {}
#         for img_file in self.image_files:
#             annotation_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt')  # предполагаем, что аннотации в .txt
#             annotation_path = os.path.join(self.data_directory, annotation_file)
#             if os.path.exists(annotation_path):
#                 # Прочитать аннотации
#                 with open(annotation_path, 'r') as f:
#                     boxes = []
#                     for line in f.readlines():
#                         # Предполагаем, что аннотации в формате: класс x_center y_center width height
#                         cls, x_center, y_center, width, height = map(float, line.strip().split())
#                         boxes.append([cls, x_center, y_center, width, height])
#                     annotations[img_file] = torch.tensor(boxes)  # Преобразуем в тензор
#             else:
#                 annotations[img_file] = torch.empty((0, 5))  # Если аннотаций нет, возвращаем пустой тензор
#         return annotations
#
#     def __len__(self):
#         # Возвращает количество изображений в наборе данных
#         return len(self.image_files)
#
#     def __getitem__(self, idx):
#         # Получает изображение и аннотации по индексу
#         img_file = self.image_files[idx]
#         img_path = os.path.join(self.data_directory, img_file)
#         image = Image.open(img_path).convert('RGB')  # Открываем изображение
#
#         # Загружаем аннотации для данного изображения
#         targets = self.annotations.get(img_file, torch.empty((0, 5)))  # Вытаскиваем аннотации, если есть
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image, targets  # Возвращаем изображение и аннотации
#
# # Пример использования
# data_directory = 'train'  # Укажите путь к вашему каталогу с данными
# transform = transforms.Compose([
#     transforms.Resize((640, 640)),  # Измените размер изображения, если требуется
#     transforms.ToTensor()  # Преобразует изображение в тензор
# ])
# dataset = CustomDataset(data_directory, transform=transform)
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
