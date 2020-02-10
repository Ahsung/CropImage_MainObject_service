import sys
import numpy as np
import json
import cv2

# json으로 받기.
inputs = sys.stdin.read()
dat = json.loads(inputs)
binary_arry = dat['binary']['data']

binary_np = np.array(binary_arry, dtype=np.uint8)

# data cv2 np convert
img_np = cv2.imdecode(binary_np, cv2.IMREAD_ANYCOLOR)

# # image change
img_np[0:50, 0:50] = 0

# convert bytes
_, imen = cv2.imencode('.jpeg', img_np)
imenb = bytes(imen)
imnb = list(imenb)

result = json.dumps({'img': imnb})
print(result)
