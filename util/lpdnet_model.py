import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from tqdm import tqdm
import gc
import os
from util.gpu_mem_track import MemTracker
import inspect

frame = inspect.currentframe()  # define a frame to track
gpu_tracker = MemTracker(frame)  # define a GPU tracker

cat_or_stack = True  # true表示cat


class LPDNet(nn.Module):
    def __init__(self, emb_dims=512, use_mFea=False, t3d=True, tfea=False, use_relu=False):
        super(LPDNet, self).__init__()
        self.negative_slope = 1e-2
        if use_relu:
            self.act_f = nn.ReLU(inplace=True)
        else:
            self.act_f = nn.LeakyReLU(negative_slope=self.negative_slope, inplace=True)
        self.use_mFea = use_mFea
        self.k = 20
        self.t3d = t3d
        self.tfea = tfea
        self.emb_dims = emb_dims
        if self.t3d:
            self.t_net3d = TranformNet(3)
        if self.tfea:
            self.t_net_fea = TranformNet(64)
        self.useBN = True
        if self.useBN:
            # [b,6,num,20] 输入 # 激活函数换成Leaky ReLU? 因为加了BN，所以bias可以舍弃
            if cat_or_stack:
                self.convDG1 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
                                             self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
                                             self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256),
                                             self.act_f)
            else:
                self.convDG1 = nn.Sequential(nn.Conv2d(64 * 1, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
                                             self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128),
                                             self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128 * 1, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256),
                                             self.act_f)
            # 在一维上进行卷积，临近也是左右概念，类似的，二维卷积，临近有上下左右的概念
            if self.use_mFea:
                self.conv1_lpd = nn.Conv1d(8, 64, kernel_size=1, bias=False)
            else:
                self.conv1_lpd = nn.Conv1d(3, 64, kernel_size=1, bias=False)
            self.conv2_lpd = nn.Conv1d(64, 64, kernel_size=1, bias=False)
            self.conv3_lpd = nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False)
            # 在relu之前进行batchNorm避免梯度消失，同时使分布不一直在变化
            self.bn1_lpd = nn.BatchNorm1d(64)
            self.bn2_lpd = nn.BatchNorm1d(64)
            self.bn3_lpd = nn.BatchNorm1d(self.emb_dims)
        else:
            # [b,6,num,20] 输入 # 激活函数换成Leaky ReLU? 因为加了BN，所以bias可以舍弃
            if cat_or_stack:
                self.convDG1 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=True), self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True), self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=True), self.act_f)
            else:
                self.convDG1 = nn.Sequential(nn.Conv2d(64 * 1, 128, kernel_size=1, bias=True), self.act_f)
                self.convDG2 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=True), self.act_f)
                self.convSN1 = nn.Sequential(nn.Conv2d(128 * 1, 256, kernel_size=1, bias=True), self.act_f)
            if self.use_mFea:
                self.conv1_lpd = nn.Conv1d(8, 64, kernel_size=1, bias=True)
            else:
                self.conv1_lpd = nn.Conv1d(3, 64, kernel_size=1, bias=True)
            self.conv2_lpd = nn.Conv1d(64, 64, kernel_size=1, bias=True)
            self.conv3_lpd = nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=True)

    # input x: # [B,1,num,num_dims]
    # output x: # [b,emb_dims,num,1]
    def forward(self, x):
        # gpu_tracker.track()
        x = torch.squeeze(x, dim=1).transpose(2, 1)  # [B,num_dims,num]
        # gpu_tracker.track()
        batch_size, num_dims, num_points = x.size()
        # 单独对坐标进行T-Net旋转
        if num_dims > 3 or self.use_mFea:
            x, feature = x.transpose(2, 1).split([3, 5], dim=2)  # [B,num,3]  [B,num,num_dims-3]
            xInit3d = x.transpose(2, 1)
            # 是否进行3D坐标旋转
            if self.t3d:
                trans = self.t_net3d(x.transpose(2, 1))
                x = torch.bmm(x, trans)
                x = torch.cat([x, feature], dim=2).transpose(2, 1)  # [B,num_dims,num]
            else:
                x = torch.cat([x, feature], dim=2).transpose(2, 1)  # [B,num_dims,num]
        else:
            xInit3d = x
            if self.t3d:
                trans = self.t_net3d(x)
                x = torch.bmm(x.transpose(2, 1), trans).transpose(2, 1)
        if self.useBN:
            x = self.act_f(self.bn1_lpd(self.conv1_lpd(x)))
            x = self.act_f(self.bn2_lpd(self.conv2_lpd(x)))
        else:
            x = self.act_f(self.conv1_lpd(x))
            x = self.act_f(self.conv2_lpd(x))

        if self.tfea:
            trans_feat = self.t_net_fea(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)

        # Serial structure
        # Danymic Graph cnn for feature space
        # gpu_tracker.track()
        if cat_or_stack:
            x = get_graph_feature(x, k=self.k)  # [b,64*2,num,20]
        else:
            x = get_graph_feature(x, k=self.k)  # [B, num_dims, num, k+1]
        # gpu_tracker.track()
        x = self.convDG1(x)  # [b,128,num,20]
        # gpu_tracker.track()
        x1 = x.max(dim=-1, keepdim=True)[0]  # [b,128,num,1]
        # gpu_tracker.track()
        x = self.convDG2(x)  # [b,128,num,20]
        # gpu_tracker.track()
        x2 = x.max(dim=-1, keepdim=True)[0]  # [b,128,num,1]
        # gpu_tracker.track()

        # Spatial Neighborhood fusion for cartesian space
        idx = knn(xInit3d, k=self.k)
        # gpu_tracker.track()
        x = get_graph_feature(x2, idx=idx, k=self.k)  # [b,128*2,num,20]
        # gpu_tracker.track()
        x = self.convSN1(x)  # [b,256,num,20]
        # gpu_tracker.track()
        x3 = x.max(dim=-1, keepdim=True)[0]  # [b,256,num,1]
        # gpu_tracker.track()

        x = torch.cat((x1, x2, x3), dim=1).squeeze(-1)  # [b,512,num]
        # gpu_tracker.track()
        if self.useBN:
            x = self.act_f(self.bn3_lpd(self.conv3_lpd(x))) # [b,emb_dims,num]
        else:
            x = self.act_f(self.conv3_lpd(x))  # [b,emb_dims,num]
        # [b,emb_dims,num]
        x = x.unsqueeze(-1)
        # [b,emb_dims,num,1]
        return x


# TranformNet
# input x [B,num_dims,num]
class TranformNet(nn.Module):
    def __init__(self, k=3, negative_slope=1e-2, use_relu=True):
        super(TranformNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        if use_relu:
            self.relu = nn.ReLU
        else:
            self.relu = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)), inplace=True)
        x = F.relu(self.bn5(self.fc2(x)), inplace=True)
        x = self.fc3(x)

        device = torch.device('cuda')

        iden = torch.eye(self.k, dtype=torch.float32, device=device).view(1, self.k * self.k).repeat(batchsize, 1)

        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


# input  [b,3,num]
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)  # [b,num,num]
    # 求坐标（维度空间）的平方和
    xx = torch.sum(x ** 2, dim=1, keepdim=True)  # [b,1,num] #x ** 2 表示点平方而不是x*x
    # 2x1x2+2y1y2+2z1z2-x1^2-y1^2-z1^2-x2^2-y2^2-z2^2=-[(x1-x2)^2+(y1-y2)^2+(z1-z2)^2]
    pairwise_distance = -xx - inner
    del inner, x
    pairwise_distance = pairwise_distance - xx.transpose(2, 1)  # [b,num,num]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


# input x [B,num_dims,num]
# output [B, num_dims*2, num, k] 领域特征tensor
def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    device = torch.device('cuda')
    # 获得索引阶梯数组
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1,
                                                               1) * num_points  # (batch_size, 1, 1) [0 num_points ... num_points*(B-1)]
    # 以batch为单位，加到索引上
    idx = idx + idx_base  # (batch_size, num_points, k)
    # 展成一维数组，方便后续索引
    idx = idx.view(-1)  # (batch_size * num_points * k)
    # 获得特征维度
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    # 改变x的shape，方便索引。被索引数组是所有batch的所有点的特征，索引数组idx为所有临近点对应的序号，从而索引出所有领域点的特征
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size * num_points * k,num_dims)
    # 统一数组形式
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    if cat_or_stack:
        # 重复k次，以便k个邻域点每个都能和中心点做运算
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # [B, num, k, num_dims]
        # 领域特征的表示，为(feature - x, x)，这种形式可以详尽参见dgcnn论文
        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2)  # [B, num_dims*2, num, k]
    else:
        x = x.view(batch_size, num_points, 1, num_dims)  # [B, num, 1, num_dims]
        # 领域特征的表示，为(feature - x, x)，这种形式可以详尽参见dgcnn论文
        feature = torch.cat((feature, x), dim=2).permute(0, 3, 1, 2)  # [B, num_dims, num, k+1]
    # del x,idx,idx_base
    return feature


class LPD(nn.Module):
    def __init__(self, args):
        super(LPD, self).__init__()
        self.emb_dims = args.emb_dims
        self.num_points = args.num_points
        self.negative_slope = 1e-2
        self.emb_nn = LPDNet(negative_slope=self.negative_slope)
        self.cycle = args.cycle

    def forward(self, *input):
        # [B,3,num]
        src = input[0]
        tgt = input[1]
        batch_size = src.size(0)
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        loss = self.getLoss(src, src_embedding, tgt_embedding)
        mse_ab_ = torch.mean((src_embedding - tgt_embedding) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ab_ = torch.mean(torch.abs(src_embedding - tgt_embedding), dim=[0, 1, 2]).item() * batch_size

        return src_embedding, tgt_embedding, loss, mse_ab_, mae_ab_

    def getLoss(self, src, src_embedding, tgt_embedding):
        batch_size, _, num_points = src.size()
        # 取k个点对做实验
        k = 32
        nk = 8
        src = src[:, :, :k]
        src_embedding_k = src_embedding[:, :, :k]
        tgt_embedding_k = tgt_embedding[:, :, :k]
        _, num_dims, _ = tgt_embedding_k.size()

        # 找到相距较远的点
        inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), src)  # [b,num,num]
        xx = torch.sum(src ** 2, dim=1, keepdim=True)  # [b,1,num] #x ** 2 表示点平方而不是x*x

        pairwise_distance = xx + inner
        pairwise_distance = pairwise_distance + xx.transpose(2, 1).contiguous()  # [b,num,num]

        idx = pairwise_distance.topk(k=nk, dim=-1)[1]  # (batch_size, k, nk)
        # 获得索引阶梯数组
        idx_base = torch.arange(0, batch_size, device=torch.device('cuda')).view(-1, 1,
                                                                                 1) * k  # (batch_size, 1, 1) [0 k ... k*(B-1)]
        # 以batch为单位，加到索引上
        idx = idx + idx_base  # (batch_size, k, nk)
        # 展成一维数组，方便后续索引
        idx = idx.view(-1)  # (batch_size * k * nk)
        # 改变x的shape，方便索引。被索引数组是所有batch的所有点的特征，索引数组idx为所有临近点对应的序号，从而索引出所有领域点的特征
        topFarTgt = tgt_embedding_k.transpose(2, 1).contiguous().view(batch_size * k, -1)[idx,
                    :]  # (batch_size * k * nk,num_dims)
        # 统一数组形式
        src_embedding_shaped = src_embedding_k.transpose(2, 1).contiguous().view(batch_size, k, 1, num_dims).repeat(
            (1, 1, nk, 1)).view(batch_size * k * nk, -1)
        tgt_embedding_shaped = tgt_embedding_k.transpose(2, 1).contiguous().view(batch_size, k, 1, num_dims).repeat(
            (1, 1, nk, 1)).view(batch_size * k * nk, -1)

        triplet_loss = nn.TripletMarginLoss(margin=0.5, p=2)
        loss_triplet = triplet_loss(src_embedding_shaped, tgt_embedding_shaped, topFarTgt)

        # 训练出的模长为1
        src_embedding = src_embedding.transpose(2, 1).contiguous()
        tgt_embedding = tgt_embedding.transpose(2, 1).contiguous()
        src_length = torch.norm(src_embedding, dim=-1)
        tgt_length = torch.norm(tgt_embedding, dim=-1)
        identity = torch.empty((batch_size, num_points), device=torch.device('cuda')).fill_(1)
        loss_norm1 = torch.sqrt(F.mse_loss(src_length, identity))
        loss_norm2 = torch.sqrt(F.mse_loss(tgt_length, identity))

        loss = loss_triplet + (loss_norm1 + loss_norm2) / 2.0 * 0.03

        return loss


def test_one_epoch(args, net, test_loader):
    net.eval()
    mse_ab = 0
    mae_ab = 0

    total_loss = 0
    num_examples = 0

    with torch.no_grad():
        for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, label in tqdm(
                test_loader):
            src = src.cuda()
            target = target.cuda()
            batch_size = src.size(0)
            num_examples += batch_size
            # [b, emb_dims, num]
            src_embedding, tgt_embedding, loss, mse_ab_, mae_ab_ = net(src, target)

            total_loss += loss.item() * batch_size
            mse_ab += mse_ab_
            mae_ab += mae_ab_

    return total_loss * 1.0 / num_examples, mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples


def train_one_epoch(args, net, train_loader, opt):
    net.train()
    mse_ab = 0
    mae_ab = 0

    total_loss = 0
    num_examples = 0

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, label in tqdm(
            train_loader):
        src = src.cuda()
        target = target.cuda()
        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size

        src_embedding, tgt_embedding, loss, mse_ab_, mae_ab_ = net(src, target)

        loss.backward()
        opt.step()

        total_loss += loss.item() * batch_size
        mse_ab += mse_ab_
        mae_ab += mae_ab_

    return total_loss * 1.0 / num_examples, mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples


def testLPD(args, net, test_loader, boardio, textio):
    test_loss, test_mse_ab, test_mae_ab = test_one_epoch(args, net, test_loader)
    test_rmse_ab = np.sqrt(test_mse_ab)

    textio.cprint('==FINAL TEST==')
    textio.cprint('A--------->B')
    textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f'
                  % (-1, test_loss, test_mse_ab, test_rmse_ab, test_mae_ab))
    return test_loss


def trainLPD(args, net, train_loader, test_loader, boardio, textio):
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(net.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # 分别在第75，150,200个epoch的时候，学习率乘以0.1
    scheduler = MultiStepLR(opt, milestones=[75, 150, 200], gamma=0.1)

    best_test_loss = np.inf

    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf

    for epoch in range(args.epochs):

        train_loss, train_mse_ab, train_mae_ab = train_one_epoch(args, net, train_loader, opt)

        test_loss, test_mse_ab, test_mae_ab = test_one_epoch(args, net, test_loader)

        scheduler.step()

        train_rmse_ab = np.sqrt(train_mse_ab)
        test_rmse_ab = np.sqrt(test_mse_ab)

        if best_test_loss >= test_loss:
            best_test_loss = test_loss
            best_test_mse_ab = test_mse_ab
            best_test_rmse_ab = test_rmse_ab
            best_test_mae_ab = test_mae_ab

            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/models/model.best.t7' % args.exp_name)

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f'
                      % (epoch, train_loss, train_mse_ab, train_rmse_ab, train_mae_ab))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f'
                      % (epoch, test_loss, test_mse_ab, test_rmse_ab, test_mae_ab))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, MSE: %f, RMSE: %f, MAE: %f'
                      % (epoch, best_test_loss, best_test_mse_ab, best_test_rmse_ab, best_test_mae_ab))

        boardio.add_scalar('A->B/train/loss', train_loss, epoch)
        boardio.add_scalar('A->B/train/MSE', train_mse_ab, epoch)
        boardio.add_scalar('A->B/train/RMSE', train_rmse_ab, epoch)
        boardio.add_scalar('A->B/train/MAE', train_mae_ab, epoch)

        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss, epoch)
        boardio.add_scalar('A->B/test/MSE', test_mse_ab, epoch)
        boardio.add_scalar('A->B/test/RMSE', test_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/MAE', test_mae_ab, epoch)

        ############BEST TEST
        boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('A->B/best_test/MSE', best_test_mse_ab, epoch)
        boardio.add_scalar('A->B/best_test/RMSE', best_test_rmse_ab, epoch)
        boardio.add_scalar('A->B/best_test/MAE', best_test_mae_ab, epoch)

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        else:
            torch.save(net.state_dict(), 'checkpoints/%s/models/model.%d.t7' % (args.exp_name, epoch))
        # for name, parameters in net.named_parameters():
        #     print(name, ':', parameters.size())
        gc.collect()
