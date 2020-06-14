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
from util.initPara import args, model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True

# os.environ['CUDA_LAUNCH_BLOCKING']="1"

LOG_FOUT = open(os.path.join(args.log_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(args) + '\n')
TOTAL_ITERATIONS = 0

BN_DECAY_DECAY_STEP = float(args.decay_step)

def get_bn_decay(batch):
    bn_momentum = cfg.BN_INIT_DECAY * \
                  (cfg.BN_DECAY_DECAY_RATE **
                   (batch * args.batch_num_queries // BN_DECAY_DECAY_STEP))
    return min(cfg.BN_DECAY_CLIP, 1 - bn_momentum)


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

# learning rate halfed every 5 epoch
def get_learning_rate(epoch):
    learning_rate = args.learning_rate * ((0.9) ** (epoch // 5))
    learning_rate = max(learning_rate, 0.00001)  # CLIP THE LEARNING RATE!
    return learning_rate


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def train():
    global model
    global HARD_NEGATIVES, TOTAL_ITERATIONS

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

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    # bn_decay = get_bn_decay(0)

    # loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
    if args.loss_function == 'quadruplet':
        # 有了第二项约束，类内间距离应该比内类距离大
        loss_function = PNV_loss.quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper
    learning_rate = get_learning_rate(0)

    train_writer = SummaryWriter(os.path.join(args.log_dir, 'train'))
    # test_writer = SummaryWriter(os.path.join(args.log_dir, 'test'))
    # while (1):
    #     a=1
    if args.optimizer == 'momentum':
        optimizer = torch.optim.SGD(
            parameters, learning_rate, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, learning_rate)
    else:
        optimizer = None
        exit(0)

    # 恢复checkpoint
    if args.resume:
        resume_filename = args.log_dir + "checkpoint.pth.tar"
        print("Resuming From ", resume_filename)
        checkpoint = torch.load(resume_filename)
        saved_state_dict = checkpoint['state_dict']
        starting_epoch = checkpoint['epoch']
        TOTAL_ITERATIONS = starting_epoch * len(TRAINING_QUERIES)

        model.load_state_dict(saved_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        starting_epoch = 0

    if not os.path.exists(args.pretrained_path):
        print("can't find pretrained model")
    else:
        print("load pretrained model")
        model.load_state_dict(torch.load(args.pretrained_path), strict=False)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        # net = torch.nn.parallel.DistributedDataParallel(net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")

    LOG_FOUT.write(cfg.cfg_str())
    LOG_FOUT.write("\n")
    LOG_FOUT.flush()

    loader_base = DataLoader(Oxford_train_base(args=args),batch_size=args.batch_num_queries, shuffle=True, drop_last=True)
    loader_advance = DataLoader(Oxford_train_advance(args=args),batch_size=args.batch_num_queries, shuffle=True, drop_last=True)

    for epoch in range(starting_epoch, args.max_epoch):
        print("epoch: ", epoch)
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(optimizer, train_writer, loss_function, epoch, loader_base, loader_advance)

        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results_' + str(epoch) + '.txt'
        eval_one_percent_recall = evaluate.evaluate_model(model)
        log_string('EVAL 1% RECALL: %s' % str(eval_one_percent_recall))

        train_writer.add_scalar("Val Recall", eval_one_percent_recall, epoch)


def train_one_epoch(optimizer, train_writer, loss_function, epoch, loader_base, loader_advance):
    global TOTAL_ITERATIONS
    global model
    model.train()
    optimizer.zero_grad()
    if epoch <= 5:
        for queries, positives, negatives, other_neg in tqdm(loader_base):
            output_queries, output_positives, output_negatives, output_other_neg = run_model(
                model, queries, positives, negatives, other_neg)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, args.margin_1,
                                 args.margin_2, use_min=args.triplet_use_best_positives, lazy=args.loss_not_lazy,
                                 ignore_zero_loss=args.loss_ignore_zero_batch)
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
            TOTAL_ITERATIONS += args.batch_num_queries
    else:
        for queries, positives, negatives, other_neg in tqdm(loader_advance):
            output_queries, output_positives, output_negatives, output_other_neg = run_model(
                model, queries, positives, negatives, other_neg)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, args.margin_1,
                                 args.margin_2, use_min=args.triplet_use_best_positives, lazy=args.loss_not_lazy,
                                 ignore_zero_loss=args.loss_ignore_zero_batch)
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
            TOTAL_ITERATIONS += args.batch_num_queries

            if (TOTAL_ITERATIONS % (1500//args.batch_num_queries*args.batch_num_queries) ==0):
                update_vectors()
                print("Updated cached feature vectors")

    if isinstance(model, nn.DataParallel):
        model_to_save = model.module
    else:
        model_to_save = model
    save_name = args.log_dir + cfg.MODEL_FILENAME + "-" + str(epoch)
    torch.save({
        'epoch': epoch,
        'iter': TOTAL_ITERATIONS,
        'state_dict': model_to_save.state_dict(),
        'optimizer': optimizer.state_dict(),
    },save_name)
    print("Model Saved As " + save_name)


def get_feature_representation(filename, model):
    model.eval()
    queries = load_pc_files([filename])
    queries = np.expand_dims(queries, axis=1)
    # if(BATCH_NUM_QUERIES-1>0):
    #    fake_queries=np.zeros((BATCH_NUM_QUERIES-1,1,NUM_POINTS,3))
    #    q=np.vstack((queries,fake_queries))
    # else:
    #    q=queries
    with torch.no_grad():
        q = torch.from_numpy(queries).float()
        q = q.to(device)
        output = model(q)
    output = output.detach().cpu().numpy()
    output = np.squeeze(output)
    model.train()
    return output


def run_model(model, queries, positives, negatives, other_neg, require_grad=True):

    feed_tensor = torch.cat(
        (queries, positives, negatives, other_neg), 1)
    feed_tensor = feed_tensor.view((-1, 1, args.num_points, 3))
    feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.to(device)
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    output = output.view(args.batch_num_queries, -1, cfg.FEATURE_OUTPUT_DIM)
    o1, o2, o3, o4 = torch.split(
        output, [1, args.positives_per_query, args.negatives_per_query, 1], dim=1)

    return o1, o2, o3, o4


if __name__ == "__main__":
    train()
