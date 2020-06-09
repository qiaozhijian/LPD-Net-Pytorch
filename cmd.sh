CUDA_VISIBLE_DEVICES=0,1 python train_pointnetvlad.py --featnet=pointnet
CUDA_VISIBLE_DEVICES=0,1 python train_pointnetvlad.py --featnet=lpdnet --model_path=./pretrained/lpd.t7