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
from util.data import TRAINING_QUERIES, device
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
    global HARD_NEGATIVES, TOTAL_ITERATIONS

    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    # 下面的type_size是4，因为我们的参数是float32也就是4B，4个字节
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    # return

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

    #loss = lazy_quadruplet_loss(q_vec, pos_vecs, neg_vecs, other_neg_vec, MARGIN1, MARGIN2)
    if args.loss_function == 'quadruplet':
        # 有了第二项约束，类内间距离应该比内类距离大
        loss_function = PNV_loss.quadruplet_loss
    else:
        loss_function = PNV_loss.triplet_loss_wrapper
    learning_rate = get_learning_rate(0)

    train_writer = SummaryWriter(os.path.join(args.log_dir, 'train'))
    #test_writer = SummaryWriter(os.path.join(args.log_dir, 'test'))
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

    for epoch in range(starting_epoch, args.max_epoch):
        print("epoch: ",epoch)
        log_string('**** EPOCH %03d ****' % (epoch))
        sys.stdout.flush()

        train_one_epoch(model, optimizer, train_writer, loss_function, epoch)

        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results_' + str(epoch) + '.txt'
        eval_one_percent_recall = evaluate.evaluate_model(model)
        log_string('EVAL 1% RECALL: %s' % str(eval_one_percent_recall))

        train_writer.add_scalar("Val Recall", eval_one_percent_recall, epoch)


def train_one_epoch(model, optimizer, train_writer, loss_function, epoch):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS

    is_training = True
    sampled_neg = 4000
    # number of hard negatives in the training tuple
    # which are taken from the sampled negatives
    hard_neg_num = args.hard_neg_per_query
    if hard_neg_num >  args.negatives_per_query:
        print("hard_neg_num >  args.negatives_per_query")
        return 

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)

    # 处理每个小batch
    for i in tqdm(range(len(train_file_idxs)//args.batch_num_queries)):
        # for i in range (5):
        # 获得一个batch的序列号
        batch_keys = train_file_idxs[i * args.batch_num_queries:(i+1)*args.batch_num_queries]
        q_tuples = []

        faulty_tuple = False
        no_other_neg = False
        for j in range(args.batch_num_queries):
            # 如果没有足够多的正样本
            if (len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < args.positives_per_query):
                faulty_tuple = True
                break
            # no cached feature vectors
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], args.positives_per_query, args.negatives_per_query,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True))
            elif (len(HARD_NEGATIVES.keys()) == 0):
                query = get_feature_representation(TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]]['negatives'][0:sampled_neg]
                # 找到离当前query最近的neg
                hard_negs = get_random_hard_negatives(query, negatives, hard_neg_num)
                # print(hard_negs)
                q_tuples.append(get_query_tuple(TRAINING_QUERIES[batch_keys[j]], args.positives_per_query, args.negatives_per_query,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
            #     如果指定了一些HARD_NEGATIVES，實際沒有
            else:
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, hard_neg_num)
                hard_negs = list(set().union(
                    HARD_NEGATIVES[batch_keys[j]], hard_negs))
                # print('hard', hard_negs)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], args.positives_per_query, args.negatives_per_query,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
            # 对点云进行增强，旋转或者加噪声
            # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

            # 这里默认使用了quadruplet loss，所以必须找到other_neg
            if (q_tuples[j][3].shape[0] != args.num_points):
                no_other_neg = True
                break

        if(faulty_tuple):
            # log_string('----' + str(i) + '-----')
            # log_string('----' + 'FAULTY TUPLE' + '-----')
            continue

        if(no_other_neg):
            # log_string('----' + str(i) + '-----')
            # log_string('----' + 'NO OTHER NEG' + '-----')
            continue

        queries = []
        positives = []
        negatives = []
        other_neg = []
        for k in range(len(q_tuples)):
            queries.append(q_tuples[k][0])
            positives.append(q_tuples[k][1])
            negatives.append(q_tuples[k][2])
            other_neg.append(q_tuples[k][3])

        queries = np.array(queries, dtype=np.float32)
        queries = np.expand_dims(queries, axis=1)
        other_neg = np.array(other_neg, dtype=np.float32)
        other_neg = np.expand_dims(other_neg, axis=1)
        positives = np.array(positives, dtype=np.float32)
        negatives = np.array(negatives, dtype=np.float32)
        # log_string('----' + str(i) + '-----')
        if (len(queries.shape) != 4):
            # log_string('----' + 'FAULTY QUERY' + '-----')
            continue

        model.train()
        optimizer.zero_grad()

        output_queries, output_positives, output_negatives, output_other_neg = run_model(
            model, queries, positives, negatives, other_neg)
        loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, args.margin_1, args.margin_2, use_min=args.triplet_use_best_positives, lazy=args.loss_not_lazy, ignore_zero_loss=args.loss_ignore_zero_batch)
        loss.backward()
        optimizer.step()

        # log_string('batch loss: %f' % loss)
        train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
        TOTAL_ITERATIONS += args.batch_num_queries

        # EVALLLL

        if (epoch > 5 and i % (1400 // args.batch_num_queries) == 29):
            TRAINING_LATENT_VECTORS = get_latent_vectors(
                model, TRAINING_QUERIES)
            print("Updated cached feature vectors")

        if (i % (6000 // args.batch_num_queries) == 101):
            if isinstance(model, nn.DataParallel):
                model_to_save = model.module
            else:
                model_to_save = model
            save_name = args.log_dir + cfg.MODEL_FILENAME

            if torch.cuda.device_count() > 1:
                torch.save({
                    'epoch': epoch,
                    'iter': TOTAL_ITERATIONS,
                    'state_dict': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                    save_name)
            else:
                torch.save({
                    'epoch': epoch,
                    'iter': TOTAL_ITERATIONS,
                    'state_dict': model_to_save.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                    save_name)

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
    queries_tensor = torch.from_numpy(queries).float()
    positives_tensor = torch.from_numpy(positives).float()
    negatives_tensor = torch.from_numpy(negatives).float()
    other_neg_tensor = torch.from_numpy(other_neg).float()
    feed_tensor = torch.cat(
        (queries_tensor, positives_tensor, negatives_tensor, other_neg_tensor), 1)
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
