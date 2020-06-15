import argparse
import sys
import evaluate
import loss.pointnetvlad_loss as PNV_loss
import models.PointNetVlad as PNV
import torch
import torch.nn as nn
from loading_pointclouds import *
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
from util.data import TRAINING_QUERIES, device, update_vectors, Oxford_train_advance, Oxford_train_base
import util.initPara as para

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True

# os.environ['CUDA_LAUNCH_BLOCKING']="1"

LOG_FOUT = open(os.path.join(para.args.log_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(para.args) + '\n')
TOTAL_ITERATIONS = 0

BN_DECAY_DECAY_STEP = float(para.args.decay_step)

def get_bn_decay(batch):
    bn_momentum = cfg.BN_INIT_DECAY * \
                  (cfg.BN_DECAY_DECAY_RATE **
                   (batch * para.args.batch_num_queries // BN_DECAY_DECAY_STEP))
    return min(cfg.BN_DECAY_CLIP, 1 - bn_momentum)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

# learning rate halfed every 5 epoch
def get_learning_rate(epoch):
    learning_rate = para.args.learning_rate * ((0.9) ** (epoch // 5))
    learning_rate = max(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def train():
    global HARD_NEGATIVES, TOTAL_ITERATIONS

    parameters = filter(lambda p: p.requires_grad, para.model.parameters())

    # bn_decay = get_bn_decay(0)

    # loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
    if para.args.loss_function == 'quadruplet':
        # 有了第二项约束，类内间距离应该比内类距离大
        loss_function = PNV_loss.quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper
    learning_rate = get_learning_rate(0)

    train_writer = SummaryWriter(os.path.join(para.args.log_dir, 'train_writer'))
    # test_writer = SummaryWriter(os.path.join(para.args.log_dir, 'test'))
    # while (1):
    #     a=1
    if para.args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(
            parameters, learning_rate, momentum=para.args.momentum)
    elif para.args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        exit(0)

    if torch.cuda.device_count() > 1:
        para.model = nn.DataParallel(para.model)
        # net = torch.nn.parallel.DistributedDataParallel(net)
        print("Let's use ", torch.cuda.device_count(), " GPUs!")

    if not os.path.exists(para.args.pretrained_path):
        print("can't find pretrained model")
    else:
        if para.args.pretrained_path[-1]=="7":
            print("load pretrained model")
            para.model.load_state_dict(torch.load(para.args.pretrained_path), strict=False)
        else:
            print("load checkpoint")
            checkpoint = torch.load(para.args.pretrained_path)
            saved_state_dict = checkpoint['state_dict']
            starting_epoch = checkpoint['epoch']
            TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)
            para.model.load_state_dict(saved_state_dict, strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            update_vectors(para.args, para.model)

    LOG_FOUT.write(cfg.cfg_str())
    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    loader_base = DataLoader(Oxford_train_base(args=para.args),batch_size=para.args.batch_num_queries, shuffle=True, drop_last=True)
    loader_advance = DataLoader(Oxford_train_advance(args=para.args),batch_size=para.args.batch_num_queries, shuffle=True, drop_last=True)

    for epoch in range(starting_epoch, para.args.max_epoch):
        print("epoch: ", epoch)
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(optimizer, train_writer, loss_function, epoch, loader_base, loader_advance)

        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results_' + str(epoch) + '.txt'
        eval_one_percent_recall = evaluate.evaluate_model(para.model)
        log_string('EVAL 1% RECALL: %s' % str(eval_one_percent_recall))

        train_writer.add_scalar("Val Recall", eval_one_percent_recall, epoch)


def train_one_epoch(optimizer, train_writer, loss_function, epoch, loader_base, loader_advance):
    global TOTAL_ITERATIONS
    para.model.train()
    optimizer.zero_grad()
    if epoch <= 5:
        for queries, positives, negatives, other_neg in tqdm(loader_base):
            output_queries, output_positives, output_negatives, output_other_neg = run_model(
                para.model, queries, positives, negatives, other_neg)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, para.args.margin_1,
                                 para.args.margin_2, use_min=para.args.triplet_use_best_positives, lazy=para.args.loss_not_lazy,
                                 ignore_zero_loss=para.args.loss_ignore_zero_batch)
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
            TOTAL_ITERATIONS += para.args.batch_num_queries
    else:
        for queries, positives, negatives, other_neg in tqdm(loader_advance):
            output_queries, output_positives, output_negatives, output_other_neg = run_model(
                para.model, queries, positives, negatives, other_neg)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, para.args.margin_1,
                                 para.args.margin_2, use_min=para.args.triplet_use_best_positives, lazy=para.args.loss_not_lazy,
                                 ignore_zero_loss=para.args.loss_ignore_zero_batch)
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
            TOTAL_ITERATIONS += para.args.batch_num_queries

            if (TOTAL_ITERATIONS % (1500//para.args.batch_num_queries*para.args.batch_num_queries) ==0):
                update_vectors(para.args, para.model)


    if isinstance(para.model, nn.DataParallel):
        model_to_save = para.model.module
    else:
        model_to_save = para.model
    save_name = para.args.model_save_path + '/' + str(epoch) + "-" + cfg.MODEL_FILENAME
    torch.save({
        'epoch': epoch,
        'iter': TOTAL_ITERATIONS,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
    },save_name)
    print("Model Saved As " + save_name)


def run_model(model, queries, positives, negatives, other_neg, require_grad=True):

    feed_tensor = torch.cat(
        (queries, positives, negatives, other_neg), 1)
    feed_tensor = feed_tensor.view((-1, 1, para.args.num_points, 3))
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device)
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    output = output.view(para.args.batch_num_queries, -1, cfg.FEATURE_OUTPUT_DIM)
    o1, o2, o3, o4 = torch.split(
        output, [1, para.args.positives_per_query, para.args.negatives_per_query, 1], dim=1)

    return o1, o2, o3, o4


if __name__ == "__main__":
    train()
