import pynvml
import os
pynvml.nvmlInit()
# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
ratio = 1024**2
while 1:
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total = meminfo.total/ratio
    used = meminfo.used/ratio
    free = meminfo.free/ratio
    print("total: ", total)
    print("used: ", used)#这里是字节bytes，所以要想得到以兆M为单位就需要除以1024**2
    print("free: ", free)#剩余显存大小
    if used < total/8:
        print("start")
        os.system('python train_pointnetvlad.py --featnet=pointnet')
        print("finish")
        break