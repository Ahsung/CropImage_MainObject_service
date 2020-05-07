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
import base64
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

#----------------------------receive stdin----------------------------
# base64 문자열으로 받기.
inputs = sys.stdin.read()

binary_arry = base64.b64decode(inputs)
binary_np = np.frombuffer(binary_arry, dtype=np.uint8)

# data cv2 np convert
im = cv2.imdecode(binary_np, cv2.IMREAD_ANYCOLOR)




## ------------------------detectron2 start!!!--------------------------------
#
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

boxes = outputs["instances"].pred_boxes.tensor
pred = outputs['instances'].pred_classes
scores = outputs["instances"].scores



#---------------------------------------------image Edit---------------------------------------------

# Get weight of importance of echo instance, and Main instance index
idx, weightlist = imageTool.get_weight(outputs, im, False)

# instances가 한개도 없다면 그대로 출력
# 있다면, 최대크기로 비율맞춰서 자른다, 단 자를 수 밖에 없는 경우는 사람을 기준으로 한다.
if weightlist.size() == torch.Size([0]):
    #없으면 그대로.
    rate16_9 = im
else:
    mx1, my1, mx2, my2 = boxes[idx]  # Main Instace box pos

    # concatenate close instace from Main_instance
    conlist = imageTool.getconInstances(boxes, idx, weightlist, 6)

    # combine img_box
    Y_S, Y_D, X_S, X_D = imageTool.combinde_img_box(boxes[conlist])

    # 사람중 가장 y축이 위에 있는 instance의 y축
    peoplelist = torch.where(pred[conlist] == 0)
    peoplelist = torch.where(pred[conlist] == 0)[0]
    if peoplelist.size() == torch.Size([0]):
        miny = Y_S
    else:
        miny = torch.min(boxes[conlist[peoplelist], 1], axis=0)
        miny = miny.values

    #비율 맞추기
    rate16_9 = imageTool.rate16_9(im,miny, Y_D, X_S, X_D)


#------------------------------print result------------------------------
# convert Base64
_, imen = cv2.imencode('.jpeg', rate16_9)
imenb = imen.tobytes()
result = base64.b64encode(imenb).decode()
print(result)
