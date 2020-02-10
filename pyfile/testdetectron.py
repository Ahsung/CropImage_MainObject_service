import json
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import torch
import sys
from imageTools import imageTool
import cv2
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()


# json으로 받기.
inputs = sys.stdin.read()
dat = json.loads(inputs)
binary_arry = dat['binary']['data']

binary_np = np.array(binary_arry, dtype=np.uint8)


# data cv2 np convert
im = cv2.imdecode(binary_np, cv2.IMREAD_ANYCOLOR)


cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
# start path in configs [dir]
fileName = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
model = model_zoo.get_config_file(fileName)

cfg.merge_from_file(model)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])

# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the detectron2:// shorthand
cfg.MODEL.WEIGHTS = "detectron2://"+model_zoo.get_weight_suffix(fileName)
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# if use ponoptic model..
#panoptic_seg, segments_info = outputs['panoptic_seg']

boxes = outputs["instances"].pred_boxes.tensor
pred = outputs['instances'].pred_classes
#masks = outputs["instances"].pred_masks
scores = outputs["instances"].scores

# Get weight of importance of echo instance, and Main instance index
idx, weightlist = imageTool.get_weight(outputs, im, False)

# if weightlist.size() == torch.Size([0]):
#     print("No instance")
#     sys.exit(0)


# concatenate close instace from Main_instance
conlist = imageTool.getconInstances(boxes, idx, weightlist, 6)

# combine img_box
Y_S, Y_D, X_S, X_D = imageTool.combinde_img_box(boxes[conlist])


# if use ponoptic model..
# v2 = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
# v2 = v2.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
# vtmp2 = v2.get_image()[:, :, ::-1]


mx1, my1, mx2, my2 = boxes[idx]  # Main Instace box pos

# 출력
# cv2.imshow('panotic',vtmp2)

result = imageTool.resize(imageTool.fitsize(im, Y_S, Y_D, X_S, X_D))
main = imageTool.resize(imageTool.fitsize(im, my1, my2, mx1, mx2))
rate16_9 = imageTool.resize(imageTool.rate16_9(im, Y_S, Y_D, X_S, X_D))

# convert bytes
_, imen = cv2.imencode('.jpeg', result)
imenb = bytes(imen)
imnb = list(imenb)

result = json.dumps({'img': imnb})
print(result)
