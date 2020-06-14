import argparse
import models.PointNetVlad as PNV
import config as cfg
import os


parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', default='log/', help='Log dir [default: log]')
parser.add_argument('--results_dir', default='results/',
                    help='results dir [default: results]')
parser.add_argument('--positives_per_query', type=int, default=2,
                    help='Number of potential positives in each training tuple [default: 2]')
parser.add_argument('--negatives_per_query', type=int, default=18,
                    help='Number of definite negatives in each training tuple [default: 18]')
parser.add_argument('--hard_neg_per_query', type=int, default=10,
                    help='Number of definite negatives in each training tuple [default: 10]')
parser.add_argument('--max_epoch', type=int, default=20,
                    help='Epoch to run [default: 20]')
parser.add_argument('--batch_num_queries', type=int, default=2,
                    help='Batch Size during training [default: 2]')
parser.add_argument('--learning_rate', type=float, default=0.000005,
                    help='Initial learning rate [default: 0.000005]')
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
parser.add_argument('--emb_dims', type=int, default=1024)

args = parser.parse_args()
cfg.DATASET_FOLDER = args.dataset_folder
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

model = PNV.PointNetVlad(feature_transform=args.fstn, num_points=args.num_points, featnet=args.featnet,
                         emb_dims=args.emb_dims)
