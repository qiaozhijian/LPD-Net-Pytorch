import torch
import torch.nn as nn
import config as cfg
import pynvml
import os
from torch.autograd import Variable
import numpy as np
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# 这里的0是GPU id
ratio = 1024**2

def print_gpu():
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = meminfo.used / ratio
    print("used: ", used)

def change(a):
    a = "dae"


import cv2
import numpy

cap = cv2.VideoCapture(0)  # 调整参数实现读取视频或调用摄像头
while 1:
    ret, frame = cap.read()
    cv2.imshow("cap", frame)
    if cv2.waitKey(delay=100) & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


if __name__ == "__main__":
    a= "{0:145,1:236}"
    change(a)
    print(a)