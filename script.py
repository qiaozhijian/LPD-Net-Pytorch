# import torch
# import torch.nn as nn
# import config as cfg
import pynvml
import os
# from torch.autograd import Variable
# import numpy as np
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
# 这里的0是GPU id
ratio = 1024**2

def print_gpu():
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    used = meminfo.used / ratio
    print("used: ", used)


from datetime import datetime
from dateutil import tz, zoneinfo

if __name__ == '__main__':
    # use timezone
    tz_sh = tz.gettz('Asia/Shanghai')
    # Shanghai timezone
    now_sh = datetime.now(tz=tz_sh)
    print(now_sh)
    now_sh = datetime.now()
    print(now_sh)