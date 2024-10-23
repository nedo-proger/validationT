import numpy as np
from tqdm import tqdm
import torch
import sys
import os
from utils.datasets import check_dataset
from utils.utils import prepare_input_tensor, non_max_suppression, box_iou


from utils.general import scale_coords, xywh2xyxy
from utils.metrics import ap_per_class



def test(data,
         model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         conf_thres: float = 0.001,
         iou_thres: float = 0.65,  # для NMS
         single_cls: bool = False,
         v5_metric: bool = False,
         names: list = None,
         num_samples: int = None
         ):
    """
    Оценка точности YOLOv7. Обрабатывает валидационный набор данных и вычисляет метрики.

    Параметры:
        model (Model): Модель YOLOv7.
        data: Данные (YAML-файл с метаданными).
        dataloader: Загрузчик данных для валидации.
        conf_thres: Порог доверия для фильтрации предсказаний.
        iou_thres: Порог IOU для NMS.
        single_cls: Использовать ли одиночный класс.
        v5_metric: Использовать ли метрику YOLOv5.
        names: Имена классов.
        num_samples: Количество образцов для тестирования.
    """

    model_output = model.output(0)
    check_dataset(data)  # Проверка данных
    nc = 1 if single_cls else int(data['nc'])  # количество классов
    iouv = torch.linspace(0.5, 0.95, 10)  # вектор IOU для mAP@0.5:0.95
    niou = iouv.numel()

    if v5_metric:
        print("Тестирование с использованием метрики AP YOLOv5...")

    seen = 0
    p, r, mp, mr, map50, map = 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    for sample_id, (img, targets, _, shapes) in enumerate(tqdm(dataloader)):
        if num_samples is not None and sample_id == num_samples:
            break
        img = prepare_input_tensor(img.numpy())
        targets = targets
        height, width = img.shape[2:]

        with torch.no_grad():
            # Выполните вывод модели
            out = torch.from_numpy(model(img)[model_output])  # вывод инференса
            # Выполните NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height])  # в пикселях

            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=None, multi_label=True)

        # Статистика по изображению
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # целевой класс
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Предсказания
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0],
                         shapes[si][1])  # предсказания в исходном пространстве
            # Назначьте все предсказания как неправильные
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device='cpu')
            if nl:
                detected = []  # индексы целей
                tcls_tensor = labels[:, 0]
                # целевые коробки
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0],
                             shapes[si][1])  # целевые метки в исходном пространстве
                # По каждому целевому классу
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # индексы предсказаний
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # индексы целей
                    # Поиск детекций
                    if pi.shape[0]:
                        # IOU предсказаний и целей
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # лучшие IOU, индексы
                        # Добавление детекций
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # обнаруженная цель
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres - это 1xn
                                if len(detected) == nl:  # все цели уже найдены в изображении
                                    break
            # Добавьте статистику (правильные, доверие, класс, цели)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Вычисление статистики
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # в numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, v5_metric=v5_metric, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # количество целей на класс
    else:
        nt = torch.zeros(1)

    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    return mp, mr, map50, map, maps, seen, nt.sum()
