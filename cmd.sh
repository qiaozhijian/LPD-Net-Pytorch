CUDA_VISIBLE_DEVICES=0,1 python train_pointnetvlad.py --batch_num_queries=6 --featnet=pointnet --eval_batch_size=20 --pretrained_path=./pretrained/pointnet.ckpt
CUDA_VISIBLE_DEVICES=0,1 python train_pointnetvlad.py --featnet=lpdnet --pretrained_path=./pretrained/lpd.t7

#python train_pointnetvlad.py --pretrained_path=./pretrained/pointnet.ckpt --eval_batch_size=20 --featnet=pointnet --eval