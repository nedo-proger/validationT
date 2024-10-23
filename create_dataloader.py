# import numpy as np
# import torch
# from tqdm import tqdm
# from utils.metrics import ap_per_class
#
#
# def validate_yolov7(data, model, dataloader, conf_thres=0.001, iou_thres=0.65, single_cls=False, names=None):
#     """
#     Validate the YOLOv7 model using the specified dataset and metrics.
#
#     Parameters:
#         data (dict): Dataset information (including number of classes).
#         model: The YOLOv7 model to evaluate.
#         dataloader: DataLoader for the validation dataset.
#         conf_thres (float): Confidence threshold for filtering predictions.
#         iou_thres (float): IoU threshold for Non-Maximum Suppression (NMS).
#         single_cls (bool): Whether to treat all classes as a single class.
#         names (list): List of class names.
#
#     Returns:
#         tuple: Precision, Recall, mAP@.5, mAP@.5:0.95, maps, number of images processed, number of labels.
#     """
#
#     model_output = model.output(0)
#     nc = 1 if single_cls else int(data['nc'])  # number of classes
#     iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
#     niou = iouv.numel()
#
#     seen = 0
#     p, r, map50, map = 0., 0., 0., 0.
#     stats = []
#
#     for sample_id, (img, targets, _, shapes) in enumerate(tqdm(dataloader)):
#         img = prepare_input_tensor(img.numpy())
#         targets[:, 2:] *= torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2]])  # Rescale targets
#
#         with torch.no_grad():
#             out = torch.from_numpy(model(img)[model_output])  # Model inference
#             out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres)
#
#         # Statistics per image
#         for si, pred in enumerate(out):
#             labels = targets[targets[:, 0] == si, 1:]  # Get the labels for the current image
#             nl = len(labels)
#
#             seen += 1
#
#             if len(pred) == 0:
#                 if nl:
#                     stats.append(
#                         (torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), labels[:, 0].tolist()))
#                 continue
#
#             predn = pred.clone()
#             scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # Scale predictions
#
#             correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device='cpu')
#             if nl:
#                 detected = []  # Detected target indices
#                 tcls_tensor = labels[:, 0]
#                 tbox = xywh2xyxy(labels[:, 1:5])  # Target boxes
#                 scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # Scale targets
#
#                 for cls in torch.unique(tcls_tensor):
#                     ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # Target indices
#                     pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # Prediction indices
#
#                     if pi.shape[0]:
#                         ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # Best IoUs
#                         detected_set = set()
#
#                         for j in (ious > iouv[0]).nonzero(as_tuple=False):
#                             d = ti[i[j]]  # Detected target
#                             if d.item() not in detected_set:
#                                 detected_set.add(d.item())
#                                 detected.append(d)
#                                 correct[pi[j]] = ious[j] > iouv  # Update correct detections
#                                 if len(detected) == nl:  # All targets detected
#                                     break
#
#             stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), labels[:, 0].tolist()))
#
#     # Compute metrics
#     stats = [np.concatenate(x, 0) for x in zip(*stats)]  # Convert to numpy arrays
#     if len(stats) and stats[0].any():
#         p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, names=names)
#         ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
#         mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
#         nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # Number of targets per class
#     else:
#         nt = torch.zeros(1)
#
#     maps = np.zeros(nc) + map
#     return mp, mr, map50, map, maps, seen, nt.sum()
#
#
# # Example of usage
# from openvino.runtime import Core
#
# core = Core()
# model_fp32 = core.read_model('model/yolov7_fp32.xml')
# compiled_model = core.compile_model(model_fp32, 'GPU')
#
# mp, mr, map50, map, maps, num_images, labels = validate_yolov7(data, compiled_model, dataloader, names=NAMES)
#
# # Print results
# s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
# print(s)
# pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # Print format
# print(pf % ('all', num_images, labels, mp, mr, map50, map))
