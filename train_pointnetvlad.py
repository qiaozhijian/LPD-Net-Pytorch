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
from util.initPara import print_gpu,log_string, BASE_LEARNING_RATE
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from util.data import TRAINING_QUERIES, device, update_vectors, Oxford_train_advance, Oxford_train_base
import util.data as datapy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True
division_epoch = 5
best_ave_one_percent_recall = 0

# os.environ['CUDA_LAUNCH_BLOCKING']="1"
TOTAL_ITERATIONS = 0

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def get_learning_rate(epoch):
    learning_rate = BASE_LEARNING_RATE*(0.922680834591**epoch)
    learning_rate = max(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def train():
    global HARD_NEGATIVES, TOTAL_ITERATIONS
    starting_epoch = 0
    ave_one_percent_recall = 0
    
    if para.args.loss_function == 'quadruplet':
        # 有了第二项约束，类内间距离应该比内类距离大
        log_string("use quadruplet_loss")
        loss_function = PNV_loss.quadruplet_loss
    else:
        log_string("use triplet_loss_wrapper")
        loss_function = PNV_loss.triplet_loss_wrapper

    if para.args.optimizer == 'momentum':
        log_string("use SGD")
        optimizer = torch.optim.SGD(para.model.parameters(), BASE_LEARNING_RATE, momentum=para.args.momentum)
    elif para.args.optimizer == 'adam':
        log_string("use adam")
        # optimizer = torch.optim.Adam(para.model.parameters(), para.args.lr, weight_decay=1e-4)
        optimizer = torch.optim.Adam(para.model.parameters(), BASE_LEARNING_RATE)
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
    # scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=1, verbose=True)

    train_writer = SummaryWriter(os.path.join(para.args.log_dir, 'train_writer'))
    # print_gpu("1")
    loader_base = DataLoader(Oxford_train_base(args=para.args),batch_size=para.args.batch_num_queries, shuffle=False, drop_last=True)
    loader_advance = DataLoader(Oxford_train_advance(args=para.args),batch_size=para.args.batch_num_queries, shuffle=False, drop_last=True)

    if starting_epoch!=0:
        log_string('EVALUATING first...')
        ave_recall, average_similarity_score, ave_one_percent_recall = evaluate.evaluate_model(para.model, tqdm_flag=True)
        log_string('EVAL %% RECALL: %s' % str(ave_one_percent_recall))
        train_writer.add_scalar("one percent recall", ave_one_percent_recall, TOTAL_ITERATIONS)

    for epoch in range(starting_epoch, para.args.max_epoch):
        log_string('**** EPOCH %03d ****' % (epoch))
        lr_temp = get_learning_rate(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_temp
        train_one_epoch(optimizer, train_writer, loss_function, epoch, loader_base, loader_advance, ave_one_percent_recall)
        # train_one_epoch_old(para.model,optimizer, train_writer, loss_function, epoch)
        log_string('EVALUATING...')
        cfg.OUTPUT_FILE = cfg.RESULTS_FOLDER + 'results_' + str(epoch) + '.txt'
        ave_recall, average_similarity_score, ave_one_percent_recall = evaluate.evaluate_model(para.model, tqdm_flag=False)
        log_string('EVAL %% RECALL: %s' % str(ave_one_percent_recall))

        save_model(epoch, optimizer, ave_one_percent_recall)
        # scheduler.step()
        # scheduler.step(ave_one_percent_recall)
        train_writer.add_scalar("Val Recall", ave_one_percent_recall, epoch)

def train_one_epoch(optimizer, train_writer, loss_function, epoch, loader_base, loader_advance, ave_one_percent_recall):
    global TOTAL_ITERATIONS

    if epoch <= division_epoch:
        for queries, positives, negatives, other_neg in tqdm(loader_base):
            para.model.train()
            optimizer.zero_grad()
            output_queries, output_positives, output_negatives, output_other_neg = run_model(
                para.model, queries, positives, negatives, other_neg)
            loss = loss_function(output_queries, output_positives, output_negatives, output_other_neg, para.args.margin_1,
                                 para.args.margin_2, use_min=para.args.triplet_use_best_positives, lazy=para.args.loss_lazy,
                                 ignore_zero_loss=para.args.loss_ignore_zero_batch)
            loss.backward()
            optimizer.step()
            train_writer.add_scalar("epoch", epoch, TOTAL_ITERATIONS)
            train_writer.add_scalar("Loss", loss.cpu().item(), TOTAL_ITERATIONS)
            train_writer.add_scalar("learn rate", optimizer.param_groups[0]['lr'], TOTAL_ITERATIONS)
            TOTAL_ITERATIONS += para.args.batch_num_queries

            if (TOTAL_ITERATIONS % ((600 * (epoch + 1)) // para.args.batch_num_queries * para.args.batch_num_queries) == 0):
                # log_string('EVALUATING...')
                ave_recall, average_similarity_score, ave_one_percent_recall = evaluate.evaluate_model(para.model, tqdm_flag=False)
                log_string('EVAL %% RECALL: %s' % str(ave_one_percent_recall), print_flag=False)
                train_writer.add_scalar("one percent recall", ave_one_percent_recall, TOTAL_ITERATIONS)

    else:
        if epoch == division_epoch + 1:
            update_vectors(para.args, para.model)
        for queries, positives, negatives, other_neg in tqdm(loader_advance):
            from time import time
            start = time()
            para.model.train()
            optimizer.zero_grad()
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
            if (TOTAL_ITERATIONS % (int(300 * (epoch + 1))//para.args.batch_num_queries*para.args.batch_num_queries) ==0):
                update_vectors(para.args, para.model, tqdm_flag=False)
            if (TOTAL_ITERATIONS % (int(600 * (epoch + 1)) // para.args.batch_num_queries * para.args.batch_num_queries) == 0):
                ave_recall, average_similarity_score, ave_one_percent_recall = evaluate.evaluate_model(para.model, tqdm_flag=False)
                log_string('EVAL %% RECALL: %s' % str(ave_one_percent_recall), print_flag=False)
                train_writer.add_scalar("one percent recall", ave_one_percent_recall, TOTAL_ITERATIONS)

def save_model(epoch, optimizer, ave_one_percent_recall):
    global best_ave_one_percent_recall
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
        'recall': ave_one_percent_recall,
    }, save_name)
    log_string("Model Saved As " + save_name)
    
    if best_ave_one_percent_recall < ave_one_percent_recall:
        best_ave_one_percent_recall = ave_one_percent_recall
        save_name = para.args.model_save_path + '/' + "best"+ "-" + cfg.MODEL_FILENAME
        torch.save({
            'epoch': epoch,
            'iter': TOTAL_ITERATIONS,
            'state_dict': model_to_save.state_dict(),
            'optimizer': optimizer.state_dict(),
            'recall': ave_one_percent_recall,
        }, save_name)
        log_string("Model Saved As " + save_name)
    

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

if __name__ == "__main__":
    if para.args.eval:
        log_string("start eval!")
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

        # log_string("model all:")
        # for name, param in para.model.named_parameters():
        #     if name=='module.point_net.stn.fc2.bias':
        #         print(param)

        ave_recall, average_similarity_score, ave_one_percent_recall = evaluate.evaluate_model(para.model, tqdm_flag=True)

        # ave_one_percent_recall = evaluate.evaluate_model2(para.model)
        print("ave_one_percent_recall: ",ave_one_percent_recall)
    else:
        train()
    print("finish")