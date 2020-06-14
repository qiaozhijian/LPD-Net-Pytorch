#!/usr/bin/env python
# -*- coding: utf-8 -*-
import glob
import os
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from loading_pointclouds import *
from sklearn.neighbors import KDTree
import torch
from util.initPara import model,args
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)
HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []


def train_one_epoch(model, optimizer, train_writer, loss_function, epoch, loader):
    global HARD_NEGATIVES
    global TRAINING_LATENT_VECTORS, TOTAL_ITERATIONS



    # 处理每个小batch
    for i in tqdm(range(len(train_file_items)//args.batch_num_queries)):
        # for i in range (5):
        # 获得一个batch的序列号
        batch_keys = train_file_items[i * args.batch_num_queries:(i+1)*args.batch_num_queries]
        q_tuples = []

        faulty_tuple = False
        no_other_neg = False
        for j in range(args.batch_num_queries):
            # 如果没有足够多的正样本
            if (len(TRAINING_QUERIES[item]["positives"]) < args.positives_per_query):
                faulty_tuple = True
                break
            # no cached feature vectors
            if (len(TRAINING_LATENT_VECTORS) == 0):
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[item], args.positives_per_query, args.negatives_per_query,
                                    TRAINING_QUERIES, hard_neg=[], other_neg=True))
            elif (len(HARD_NEGATIVES.keys()) == 0):
                query = get_feature_representation(TRAINING_QUERIES[item]['query'], model)
                random.shuffle(TRAINING_QUERIES[item]['negatives'])
                negatives = TRAINING_QUERIES[item]['negatives'][0:sampled_neg]
                # 找到离当前query最近的neg
                hard_negs = get_random_hard_negatives(query, negatives, hard_neg_num)
                # print(hard_negs)
                q_tuples.append(get_query_tuple(TRAINING_QUERIES[item], args.positives_per_query, args.negatives_per_query,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
            #     如果指定了一些HARD_NEGATIVES，實際沒有
            else:
                query = get_feature_representation(
                    TRAINING_QUERIES[item]['query'], model)
                random.shuffle(TRAINING_QUERIES[item]['negatives'])
                negatives = TRAINING_QUERIES[item
                                             ]['negatives'][0:sampled_neg]
                hard_negs = get_random_hard_negatives(
                    query, negatives, hard_neg_num)
                hard_negs = list(set().union(
                    HARD_NEGATIVES[item], hard_negs))
                # print('hard', hard_negs)
                q_tuples.append(
                    get_query_tuple(TRAINING_QUERIES[item], args.positives_per_query, args.negatives_per_query,
                                    TRAINING_QUERIES, hard_negs, other_neg=True))
            # 对点云进行增强，旋转或者加噪声
            # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[item],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
            # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[item],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

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

        for queries, positives, negatives, other_neg in tqdm(loader):
            if queries==0:
                continue
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

def get_random_hard_negatives(query_vec, random_negs, hard_neg_num):
    global TRAINING_LATENT_VECTORS

    latent_vecs = []
    for j in range(len(random_negs)):
        latent_vecs.append(TRAINING_LATENT_VECTORS[random_negs[j]])

    latent_vecs = np.array(latent_vecs)
    nbrs = KDTree(latent_vecs)
    distances, indices = nbrs.query(np.array([query_vec]), k=hard_neg_num)
    hard_negs = np.squeeze(np.array(random_negs)[indices[0]])
    hard_negs = hard_negs.tolist()
    return hard_negs

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

# 设置成随机抽取的
class Oxford_train_base(Dataset):
    def __init__(self, args):
        self.num_points = args.args
        self.positives_per_query=args.positives_per_query
        self.negatives_per_query=args.negatives_per_query
        self.train_len=len(TRAINING_QUERIES.keys())
        # self.train_file_items = np.random.permutation(np.arange(0, self.train_len))
        # self.train_file_items = np.arange(0, self.train_len)
        print('Load Oxford Dataset')
        self.data, self.label = []

    def __getitem__(self, item):
        if (len(TRAINING_QUERIES[item]["positives"]) < self.positives_per_query):
            return 0, 0, 0, 0
        # no cached feature vectors
        q_tuples=get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                TRAINING_QUERIES, hard_neg=[], other_neg=True)
        
        # 对点云进行增强，旋转或者加噪声
        # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[item],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
        # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[item],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

        # 这里默认使用了quadruplet loss，所以必须找到other_neg
        if (q_tuples[3].shape[0] != self.num_points):
            print('----' + 'FAULTY other_neg' + '-----')
            return 0, 0, 0, 0

        queries = np.expand_dims(np.array(q_tuples[0], dtype=np.float32), axis=1)
        other_neg = np.expand_dims(np.array(q_tuples[3], dtype=np.float32), axis=1)
        positives = np.array(q_tuples[1], dtype=np.float32)
        negatives = np.array(q_tuples[2], dtype=np.float32)

        if (len(queries.shape) != 4):
            print('----' + 'FAULTY QUERY' + '-----')
            return 0, 0, 0, 0

        return queries, positives, negatives, other_neg

    def __len__(self):
        return self.train_len

def update_vectors():
    global TRAINING_LATENT_VECTORS
    global model,args
    TRAINING_LATENT_VECTORS = get_latent_vectors(model, TRAINING_QUERIES)

class Oxford_train_advance(Dataset):
    def __init__(self, args):
        self.num_points = args.args
        self.positives_per_query = args.positives_per_query
        self.negatives_per_query = args.negatives_per_query
        self.train_len = len(TRAINING_QUERIES.keys())
        # self.train_file_items = np.random.permutation(np.arange(0, self.train_len))
        # self.train_file_items = np.arange(0, self.train_len)
        print('Load Oxford Dataset')
        self.data, self.label = []
        self.sampled_neg = 4000
        self.hard_neg_num = args.hard_neg_per_query
        if self.hard_neg_num > args.negatives_per_query:
            print("self.hard_neg_num >  args.negatives_per_query")

    def __getitem__(self, item):
        global model
        if (len(TRAINING_QUERIES[item]["positives"]) < self.positives_per_query):
            return 0, 0, 0, 0
        if (len(HARD_NEGATIVES.keys()) == 0):
            query = get_feature_representation(TRAINING_QUERIES[item]['query'], model)
            random.shuffle(TRAINING_QUERIES[item]['negatives'])
            negatives = TRAINING_QUERIES[item]['negatives'][0:self.sampled_neg]
            # 找到离当前query最近的neg
            hard_negs = get_random_hard_negatives(query, negatives, self.hard_neg_num)
            # print(hard_negs)
            q_tuples=get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                            TRAINING_QUERIES, hard_negs, other_neg=True)
        #     如果指定了一些HARD_NEGATIVES，實際沒有
        else:
            query = get_feature_representation(
                TRAINING_QUERIES[item]['query'], model)
            random.shuffle(TRAINING_QUERIES[item]['negatives'])
            negatives = TRAINING_QUERIES[item
                        ]['negatives'][0:self.sampled_neg]
            hard_negs = get_random_hard_negatives(
                query, negatives, self.hard_neg_num)
            hard_negs = list(set().union(
                HARD_NEGATIVES[item], hard_negs))
            # print('hard', hard_negs)
            q_tuples=get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                TRAINING_QUERIES, hard_negs, other_neg=True)

        # 这里默认使用了quadruplet loss，所以必须找到other_neg
        if (q_tuples[3].shape[0] != self.num_points):
            print('----' + 'FAULTY other_neg' + '-----')
            return 0, 0, 0, 0

        queries = np.expand_dims(np.array(q_tuples[0], dtype=np.float32), axis=1)
        other_neg = np.expand_dims(np.array(q_tuples[3], dtype=np.float32), axis=1)
        positives = np.array(q_tuples[1], dtype=np.float32)
        negatives = np.array(q_tuples[2], dtype=np.float32)

        if (len(queries.shape) != 4):
            print('----' + 'FAULTY QUERY' + '-----')
            return 0, 0, 0, 0

        return queries, positives, negatives, other_neg

    def __len__(self):
        return self.train_len

def get_latent_vectors(model, dict_to_process):
    global args
    train_file_idxs = np.arange(0, len(dict_to_process.keys()))

    batch_num = args.batch_num_queries * \
        (1 + args.positives_per_query + args.negatives_per_query + 1)
    q_output = []

    model.eval()

    for q_index in range(len(train_file_idxs)//batch_num):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index+1)*(batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(dict_to_process[index]["query"])
        queries = load_pc_files(file_names)

        feed_tensor = torch.from_numpy(queries).float()
        feed_tensor = feed_tensor.unsqueeze(1)
        feed_tensor = feed_tensor.to(device)
        with torch.no_grad():
            out = model(feed_tensor)

        out = out.detach().cpu().numpy()
        out = np.squeeze(out)

        q_output.append(out)

    q_output = np.array(q_output)
    if(len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in range((len(train_file_idxs) // batch_num * batch_num), len(dict_to_process.keys())):
        index = train_file_idxs[q_index]
        queries = load_pc_files([dict_to_process[index]["query"]])
        queries = np.expand_dims(queries, axis=1)

        # if (BATCH_NUM_QUERIES - 1 > 0):
        #    fake_queries = np.zeros((BATCH_NUM_QUERIES - 1, 1, NUM_POINTS, 3))
        #    q = np.vstack((queries, fake_queries))
        # else:
        #    q = queries

        #fake_pos = np.zeros((BATCH_NUM_QUERIES, POSITIVES_PER_QUERY, NUM_POINTS, 3))
        #fake_neg = np.zeros((BATCH_NUM_QUERIES, NEGATIVES_PER_QUERY, NUM_POINTS, 3))
        #fake_other_neg = np.zeros((BATCH_NUM_QUERIES, 1, NUM_POINTS, 3))
        #o1, o2, o3, o4 = run_model(model, q, fake_pos, fake_neg, fake_other_neg)
        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            o1 = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output))
        else:
            q_output = output

    model.train()
    # print(q_output.shape)
    # return q_output