import sys
import evaluate
import loss.pointnetvlad_loss as PNV_loss
import torch
import torch.nn as nn
from loading_pointclouds import *
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
import util.initPara as para
from util.initPara import print_gpu
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from util.data import TRAINING_QUERIES, device, update_vectors, Oxford_train_advance, Oxford_train_base

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True
division_epoch = 5

# os.environ['CUDA_LAUNCH_BLOCKING']="1"

LOG_FOUT = open(os.path.join(para.args.log_dir, 'log_train.txt'), 'w')
LOG_FOUT.write(str(para.args) + '\n')
LOG_FOUT.flush()
TOTAL_ITERATIONS = 0


def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True


def train():
    global HARD_NEGATIVES, TOTAL_ITERATIONS
    starting_epoch = 0
    eval_one_percent_recall = 0

    if para.args.loss_function == 'quadruplet':
        # 有了第二项约束，类内间距离应该比内类距离大
        log_string("use quadruplet_loss")
        loss_function = PNV_loss.quadruplet_loss
    else:
        log_string("use triplet_loss_wrapper")
        loss_function = PNV_loss.triplet_loss_wrapper

    if para.args.optimizer == 'momentum':
        log_string("use SGD")
        optimizer = torch.optim.SGD(para.model.parameters(), para.args.lr, momentum=para.args.momentum)
    elif para.args.optimizer == 'adam':
        log_string("use adam")
        optimizer = torch.optim.Adam(para.model.parameters(), para.args.lr, weight_decay=1e-4)
    else:
        log_string("optimizer None")
        optimizer = None
        exit(0)
    #
    # print_gpu("0")
    if not os.path.exists(para.args.pretrained_path):
        log_string("can't find pretrained model" + para.args.pretrained_path)
    else:
        if para.args.pretrained_path[-1]=="7":
            log_string("load pretrained model" + para.args.pretrained_path)
            para.model.load_state_dict(torch.load(para.args.pretrained_path), strict=True)
        else:
            checkpoint = torch.load(para.args.pretrained_path)
            saved_state_dict = checkpoint['state_dict']
            starting_epoch = checkpoint['epoch'] + 1
            TOTAL_ITERATIONS = checkpoint['iter']
            para.model.load_state_dict(saved_state_dict, strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            log_string("load checkpoint" + para.args.pretrained_path+ " starting_epoch: "+ str(starting_epoch))
    if torch.cuda.device_count() > 1:
        para.model = nn.DataParallel(para.model)
        log_string("Let's use "+ str(torch.cuda.device_count())+ " GPUs!")
    if starting_epoch > division_epoch + 1:
        update_vectors(para.args, para.model)

    # scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    # recall 停止上升时
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True)

    train_writer = SummaryWriter(os.path.join(para.args.log_dir, 'train_writer'))
    # print_gpu("1")
    loader_base = DataLoader(Oxford_train_base(args=para.args),batch_size=para.args.batch_num_queries, shuffle=True, drop_last=True)
    loader_advance = DataLoader(Oxford_train_advance(args=para.args),batch_size=para.args.batch_num_queries, shuffle=True, drop_last=True)

    log_string('EVALUATING first...')
    eval_one_percent_recall = evaluate.evaluate_model(para.model)
    log_string('EVAL %% RECALL: %s' % str(eval_one_percent_recall))
    train_writer.add_scalar("one percent recall", eval_one_percent_recall, TOTAL_ITERATIONS)

    for epoch in range(starting_epoch, para.args.max_epoch):
        log_string('**** EPOCH %03d ****' % (epoch))
        # train_one_epoch(optimizer, train_writer, loss_function, epoch, loader_base, loader_advance, eval_one_percent_recall)
        train_one_epoch_old(para.model,optimizer, train_writer, loss_function, epoch)
        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results_' + str(epoch) + '.txt'
        eval_one_percent_recall = evaluate.evaluate_model(para.model)
        log_string('EVAL %% RECALL: %s' % str(eval_one_percent_recall))

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
            'recall': eval_one_percent_recall,
        }, save_name)
        log_string("Model Saved As " + save_name)

        # scheduler.step()
        scheduler.step(eval_one_percent_recall)

        train_writer.add_scalar("Val Recall", eval_one_percent_recall, epoch)


def train_one_epoch(optimizer, train_writer, loss_function, epoch, loader_base, loader_advance, eval_one_percent_recall):
    global TOTAL_ITERATIONS
    para.model.train()
    optimizer.zero_grad()

    if epoch <= division_epoch:
        for queries, positives, negatives, other_neg in tqdm(loader_base):
            output_queries, output_positives, output_negatives, output_other_neg = run_model(
                para.model, queries, positives, negatives, other_neg)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, para.args.margin_1,
                                 para.args.margin_2, use_min=para.args.triplet_use_best_positives, lazy=para.args.loss_not_lazy,
                                 ignore_zero_loss=para.args.loss_ignore_zero_batch)
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("epoch", epoch, TOTAL_ITERATIONS)
            train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
            train_writer.add_scalar("learn rate", optimizer.param_groups[0]['lr'], TOTAL_ITERATIONS)
            TOTAL_ITERATIONS += para.args.batch_num_queries

            if (TOTAL_ITERATIONS % (3600 // para.args.batch_num_queries * para.args.batch_num_queries) == 0):
                # log_string('EVALUATING...')
                eval_one_percent_recall = evaluate.evaluate_model(para.model,save_flag=False)
                log_string('EVAL %% RECALL: %s' % str(eval_one_percent_recall))
                train_writer.add_scalar("one percent recall", eval_one_percent_recall, TOTAL_ITERATIONS)

    else:
        if epoch == division_epoch + 1:
            update_vectors(para.args, para.model)
        for queries, positives, negatives, other_neg in tqdm(loader_advance):
            from time import time
            start = time()
            output_queries, output_positives, output_negatives, output_other_neg = run_model(
                para.model, queries, positives, negatives, other_neg)
            # log_string("train: ",time()-start)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, para.args.margin_1,
                                 para.args.margin_2, use_min=para.args.triplet_use_best_positives, lazy=para.args.loss_lazy,
                                 ignore_zero_loss=para.args.loss_ignore_zero_batch)
            # log_string("train: ",time()-start)
            # 比较耗时
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("epoch", epoch, TOTAL_ITERATIONS)
            train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
            train_writer.add_scalar("learn rate", optimizer.param_groups[0]['lr'], TOTAL_ITERATIONS)
            TOTAL_ITERATIONS += para.args.batch_num_queries
            # log_string("train: ",time()-start)
            if (TOTAL_ITERATIONS % (int(1200 + 300 * (epoch-division_epoch))//para.args.batch_num_queries*para.args.batch_num_queries) ==0):
                update_vectors(para.args, para.model)
            if (TOTAL_ITERATIONS % (3600 // para.args.batch_num_queries * para.args.batch_num_queries) == 0):
                # log_string('EVALUATING...')
                eval_one_percent_recall = evaluate.evaluate_model(para.model,save_flag=False)
                log_string('EVAL %% RECALL: %s' % str(eval_one_percent_recall))
                train_writer.add_scalar("one percent recall", eval_one_percent_recall, TOTAL_ITERATIONS)


def run_model(model, queries, positives, negatives, other_neg, require_grad=True):
    # print_gpu("2")
    feed_tensor = torch.cat((queries, positives, negatives, other_neg), 1)
    feed_tensor = feed_tensor.view((-1, 1, para.args.num_points, 3))
    # feed_tensor.requires_grad_(require_grad)
    feed_tensor = feed_tensor.cuda()
    # print_gpu("3")
    if require_grad:
        output = model(feed_tensor)
    else:
        with torch.no_grad():
            output = model(feed_tensor)
    output = output.view(para.args.batch_num_queries, -1, cfg.FEATURE_OUTPUT_DIM)
    o1, o2, o3, o4 = torch.split(
        output, [1, para.args.positives_per_query, para.args.negatives_per_query, 1], dim=1)
    return o1, o2, o3, o4

def train_one_epoch_old(model, optimizer, train_writer, loss_function, epoch):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS
    from util.data import get_feature_representation,get_query_tuple,get_random_hard_negatives
    from evaluate import get_latent_vectors
    is_training = True
    sampled_neg = 4000
    # number of hard negatives in the training tuple
    # which are taken from the sampled negatives
    num_to_take = 10

    # Shuffle train files
    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))
    np.random.shuffle(train_file_idxs)

    for i in range(len(train_file_idxs)//para.args.batch_num_queries):
        # for i in range (5):
        batch_keys = train_file_idxs[i *
                                     para.args.batch_num_queries:(i+1)*para.args.batch_num_queries]
        q_tuples = []

        faulty_tuple = False
        no_other_neg = False
        for j in range(para.args.batch_num_queries):
            if (len(TRAINING_QUERIES[batch_keys[j]]["positives"]) < para.args.positives_per_query):
                faulty_tuple = True
                break

            # no cached feature vectors
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], para.args.positives_per_query, para.args.negatives_per_query,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_neg=[], other_neg=True))

            elif (len(HARD_NEGATIVES.keys()) == 0):
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                print(hard_negs)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], para.args.positives_per_query, para.args.negatives_per_query,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            else:
                query = get_feature_representation(
                    TRAINING_QUERIES[batch_keys[j]]['query'], model)
                random.shuffle(TRAINING_QUERIES[batch_keys[j]]['negatives'])
                negatives = TRAINING_QUERIES[batch_keys[j]
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, num_to_take)
                hard_negs = list(set().union(
                    HARD_NEGATIVES[batch_keys[j]], hard_negs))
                print('hard', hard_negs)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[batch_keys[j]], para.args.positives_per_query, para.args.negatives_per_query,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
                # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[batch_keys[j]],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

            if (q_tuples[j][3].shape[0] != cfg.NUM_POINTS):
                no_other_neg = True
                break

        if(faulty_tuple):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'FAULTY TUPLE' + '-----')
            continue

        if(no_other_neg):
            log_string('----' + str(i) + '-----')
            log_string('----' + 'NO OTHER NEG' + '-----')
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
        log_string('----' + str(i) + '-----')
        if (len(queries.shape) != 4):
            log_string('----' + 'FAULTY QUERY' + '-----')
            continue

        model.train()
        optimizer.zero_grad()

        output_queries, output_positives, output_negatives, output_other_neg = run_model(
            model, queries, positives, negatives, other_neg)
        loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, cfg.MARGIN_1, cfg.MARGIN_2, use_min=False, lazy=True, ignore_zero_loss=False)
        loss.backward()
        optimizer.step()

        # log_string('batch loss: %f' % loss)
        train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
        TOTAL_ITERATIONS += para.args.batch_num_queries

        # EVALLLL

        if (epoch > 5 and i % (1400 // para.args.batch_num_queries) == 29):
            TRAINING_LATENT_VECTORS = get_latent_vectors(
                model, TRAINING_QUERIES)
            print("Updated cached feature vectors")

        # if (TOTAL_ITERATIONS % (1000 // para.args.batch_num_queries * para.args.batch_num_queries) == 0):
        #     if isinstance(para.model, nn.DataParallel):
        #         model_to_save = model.module
        #     else:
        #         model_to_save = model
        #     save_name = para.args.model_save_path + '/' + str(epoch) + "-" + cfg.MODEL_FILENAME
        #     torch.save({
        #         'epoch': epoch,
        #         'iter': TOTAL_ITERATIONS,
        #         'state_dict': model_to_save.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, save_name)
        #     log_string("Model Saved As " + save_name)


if __name__ == "__main__":
    if para.args.eval:
        log_string("start eval!")
        #
        # print_gpu("0")
        # ave_one_percent_recall = evaluate.evaluate_model(para.model)
        # print("ave_one_percent_recall: ",ave_one_percent_recall)
        if not os.path.exists(para.args.pretrained_path):
            log_string("can't find pretrained model" + para.args.pretrained_path)
        else:
            if para.args.pretrained_path[-1] == "7":
                log_string("load pretrained model" + para.args.pretrained_path)
                para.model.load_state_dict(torch.load(para.args.pretrained_path), strict=True)
            else:
                checkpoint = torch.load(para.args.pretrained_path)
                saved_state_dict = checkpoint['state_dict']
                starting_epoch = checkpoint['epoch'] + 1
                TOTAL_ITERATIONS = checkpoint['iter']
                para.model.load_state_dict(saved_state_dict, strict=True)
                log_string("load checkpoint success" + para.args.pretrained_path+" starting_epoch: "+str(starting_epoch))
        # 加载网络参数需要在并行之前，因为并行会加“module”
        if torch.cuda.device_count() > 1:
            para.model = nn.DataParallel(para.model)
            log_string("Let's use "+ str(torch.cuda.device_count())+ " GPUs!")
        ave_one_percent_recall = evaluate.evaluate_model(para.model,save_flag=False)
        print("ave_one_percent_recall: ",ave_one_percent_recall)
    else:
        train()
    print("finish")