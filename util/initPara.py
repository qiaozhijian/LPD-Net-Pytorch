import argparse
import os
from datetime import datetime
import torch
import numpy as np
import config as cfg
import util.PointNetVlad as PNV
import pynvml
from dateutil import tz

pynvml.nvmlInit()
handle0 = pynvml.nvmlDeviceGetHandleByIndex(0)
if torch.cuda.device_count() > 1:
    handle1 = pynvml.nvmlDeviceGetHandleByIndex(1)
ratio = 1024 ** 2

def print_gpu(s=""):
    if torch.cuda.device_count() > 1:
        meminfo0 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
        meminfo1 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
        used = (meminfo0.used + meminfo1.used) / ratio
    else:
        meminfo0 = pynvml.nvmlDeviceGetMemoryInfo(handle0)
        used = meminfo0.used / ratio
    print(s+" used: ", used)

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', default='results/',
                    help='results dir [default: results]')
parser.add_argument('--positives_per_query', type=int, default=2,
                    help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=18,
                    help='Number of definite negatives in each training tuple [default: 18]')
parser.add_argument('--hard_neg_per_query', type=int, default=10,
                    help='Number of definite negatives in each training tuple [default: 10]')
parser.add_argument('--max_epoch', type=int, default=30,
                    help='Epoch to run [default: 20]')
parser.add_argument('--eval_batch_size', type=int, default=6,
                    help='test Batch Size during training [default: 6]')
parser.add_argument('--batch_num_queries', type=int, default=2,
                    help='Batch Size during training [default: 2]')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--num_points', type=int, default=4096,
                    help='num_points [default: 4096]')
parser.add_argument('--decay_step', type=int, default=200000,
                    help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7,
                    help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--margin_1', type=float, default=0.5,
                    help='Margin for hinge loss [default: 0.5]')
parser.add_argument('--margin_2', type=float, default=0.2,
                    help='Margin for hinge loss [default: 0.2]')
parser.add_argument('--loss_function', default='quadruplet', choices=[
    'triplet', 'quadruplet'], help='triplet or quadruplet [default: quadruplet]')
parser.add_argument('--loss_not_lazy', action='store_false',
                    help='If present, do not use lazy variant of loss')
parser.add_argument('--loss_ignore_zero_batch', action='store_true',
                    help='If present, mean only batches with loss > 0.0')
parser.add_argument('--triplet_use_best_positives', action='store_true',
                    help='If present, use best positives, otherwise use hardest positives')
parser.add_argument('--resume', action='store_true',
                    help='If present, restore checkpoint and resume training')
parser.add_argument('--dataset_folder', default='./benchmark_datasets/',
                    help='PointNetVlad Dataset Folder')
parser.add_argument('--pretrained_path', type=str, default='', metavar='N',
                    help='Pretrained model path')
parser.add_argument('--featnet', type=str, default='lpdnet', metavar='N',
                    help='feature net')
parser.add_argument('--fstn', action='store_true', default=False,
                    help='feature transform')
parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate (min: 0.00001, 0.1 if using sgd)')
parser.add_argument('--emb_dims', type=int, default=1024)
parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
parser.add_argument('--log_dir', default='checkpoints/', help='Log dir [default: log]')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
args = parser.parse_args()

torch.backends.cudnn.deterministic = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

cfg.DATASET_FOLDER = args.dataset_folder
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

model = PNV.PointNetVlad(feature_transform=args.fstn, num_points=args.num_points, featnet=args.featnet,
                         emb_dims=args.emb_dims)

if args.eval:
    file = args.pretrained_path
    filename = os.path.basename(file)
    filename = os.path.splitext(filename)[0]
    tz_sh = tz.gettz('Asia/Shanghai')
    args.exp_name = filename + '-' + datetime.now(tz=tz_sh).strftime("%d-%H-%M-%S") + '-test'
else:
    tz_sh = tz.gettz('Asia/Shanghai')
    args.exp_name = args.featnet + '-' + datetime.now(tz=tz_sh).strftime("%d-%H-%M-%S")
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
if not os.path.exists('checkpoints/' + args.exp_name):
    os.makedirs('checkpoints/' + args.exp_name)
if not os.path.exists('checkpoints/' + args.exp_name + '/' + 'models'):
    os.makedirs('checkpoints/' + args.exp_name + '/' + 'models')
args.model_save_path = 'checkpoints/' + args.exp_name + '/' + 'models'
args.log_dir = 'checkpoints/' + args.exp_name
cfg.RESULTS_FOLDER = args.log_dir + '/' + cfg.RESULTS_FOLDER
if not os.path.exists(cfg.RESULTS_FOLDER):
    os.makedirs(cfg.RESULTS_FOLDER)

para = sum([np.prod(list(p.size())) for p in model.parameters()])
# 下面的type_size是4，因为我们的参数是float32也就是4B，4个字节
print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))

# 知乎说会节省显存，没啥用
# model.apply(inplace_relu)

if torch.cuda.is_available():
    model = model.cuda()
    print("use cuda!")
else:
    print("use cpu...")
    model = model.cpu()

# print("model all:")
# for name, param in model.named_parameters():
#     print(name)

# 一般情况下模型的requires_grad都为true
# parameters = filter(lambda p: p.requires_grad, para.model.parameters())
# parameters2 = filter(lambda p: True, para.model.parameters())
# import operator
# print(operator.eq(list(parameters),list(parameters2)))
# print(set(list(parameters)).difference(set(list(parameters2))))


