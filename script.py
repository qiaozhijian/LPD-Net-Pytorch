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


if __name__ == "__main__":
    train_file_idxs = np.arange(0,100)
    file_indices = train_file_idxs[5:8]
    print(file_indices)
    print(list(range(5,8)))
    learning_rate = 0.000005 * ((0.9) ** (0 // 5))
    learning_rate = max(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    # 1e-05
    print(learning_rate)