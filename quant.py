import os

model_path = 'model1/yolov7_fp32.xml'
if not os.path.exists(model_path):
    print(f"Модель не найдена: {model_path}")


