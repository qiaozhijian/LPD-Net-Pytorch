import torch
import torch.nn as nn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        print("use cuda!")
    else:
        print("use cpu...")
    if torch.cuda.device_count() > 0:
        print("Let's use", torch.cuda.device_count(), "GPUs!")