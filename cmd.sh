CUDA_VISIBLE_DEVICES=0,1 python train_pointnetvlad.py --batch_num_queries=6 --featnet=pointnet --eval_batch_size=20 --pretrained_path=./pretrained/pointnet.ckpt
CUDA_VISIBLE_DEVICES=0,1 python train_pointnetvlad.py --featnet=lpdnet --pretrained_path=./pretrained/lpd.t7

#21708 一个epoch大概这么多样本
#python train_pointnetvlad.py --pretrained_path=./pretrained/pointnet.ckpt --eval_batch_size=20 --featnet=pointnet --eval

#--featnet=pointnet
#--batch_num_queries=5
#--pretrained_path=./checkpoints/originloadfast/0.ckpt
#--eval
#--eval_batch_size=2
#--fstn

#1024 0.001学习率,发散

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_pointnetvlad.py --batch_num_queries=2 --eval_batch_size=5 --fstn --positives_per_query=2 --negatives_per_query=10 --hard_neg_per_query=6 --emb_dims=512