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
import util.initPara as para
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load dictionary of training queries
TRAINING_QUERIES = get_queries_dict(cfg.TRAIN_FILE)
TEST_QUERIES = get_queries_dict(cfg.TEST_FILE)
HARD_NEGATIVES = {}
TRAINING_LATENT_VECTORS = []
TRAINING_POINT_CLOUD = []

for i in tqdm(range(len(TRAINING_QUERIES))):
    filename = TRAINING_QUERIES[i]["query"]
    pc = load_pc_file(filename)
    TRAINING_POINT_CLOUD.append(pc)
TRAINING_POINT_CLOUD = np.asarray(TRAINING_POINT_CLOUD).reshape(-1,4096,3)

def flat(l):
    for k in l:
        if not isinstance(k, (list, tuple)):
            yield k
        else:
            yield from flat(k)

def get_query_tuple_fast(item, dict_value, num_pos, num_neg, QUERY_DICT, hard_neg=[], other_neg=False):
    # get query tuple for dictionary entry
    # return list [query,positives,negatives]
    start = time()
    query = TRAINING_POINT_CLOUD[item] # 就一个
    random.shuffle(dict_value["positives"])
    # 不必考虑正样本是否充足，因为之前判断过
    positives = TRAINING_POINT_CLOUD[(dict_value["positives"][:num_pos])]

    neg_indices = []
    if(len(hard_neg) == 0):
        random.shuffle(dict_value["negatives"])
        neg_indices=dict_value["negatives"][:num_neg]
    else:
        neg_indices.append(hard_neg)
        j = 0
        # 如果hard不够，再进行补充
        while(len(neg_indices) < num_neg):
            if not dict_value["negatives"][j] in hard_neg:
                neg_indices.append(dict_value["negatives"][j])
            j += 1
    neg_indices = list(flat(neg_indices))
    negatives = TRAINING_POINT_CLOUD[neg_indices]

    # print("load time: ",time()-start)
    # 是否需要额外的neg（Quadruplet loss需要）
    if other_neg is False:
        return [query, positives, negatives]
    else:
        # get neighbors of negatives and query
        neighbors = []
        for pos in dict_value["positives"]:
            neighbors.append(pos)
        for neg in neg_indices:
            for pos in QUERY_DICT[neg]["positives"]:
                neighbors.append(pos)
        # 减去与neighbors公共有的部分，剩下既不进也不远的那些部分
        neighbors = list(flat(neighbors))
        possible_negs = list(set(QUERY_DICT.keys())-set(neighbors))
        random.shuffle(possible_negs)
        if(len(possible_negs) == 0):
            return [query, positives, negatives, np.array([])]
        neg2 = TRAINING_POINT_CLOUD[possible_negs[0]] # 就一个

        return [query, positives, negatives, neg2]

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
        self.num_points = args.num_points
        self.positives_per_query=args.positives_per_query
        self.negatives_per_query=args.negatives_per_query
        self.train_len=len(TRAINING_QUERIES.keys())
        # self.train_file_items = np.random.permutation(np.arange(0, self.train_len))
        # self.train_file_items = np.arange(0, self.train_len)
        print('Load Oxford Dataset')
        # self.data, self.label = []
        self.last = []
    def __getitem__(self, item):
        if (len(TRAINING_QUERIES[item]["positives"]) < self.positives_per_query):
            if self.last==[]:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]
        # no cached feature vectors
        # q_tuples=get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
        #                         TRAINING_QUERIES, hard_neg=[], other_neg=True)
        q_tuples=get_query_tuple_fast(item, TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                TRAINING_QUERIES, hard_neg=[], other_neg=True)
        # 对点云进行增强，旋转或者加噪声
        # q_tuples.append(get_rotated_tuple(TRAINING_QUERIES[item],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))
        # q_tuples.append(get_jittered_tuple(TRAINING_QUERIES[item],POSITIVES_PER_QUERY,NEGATIVES_PER_QUERY, TRAINING_QUERIES, hard_negs, other_neg=True))

        # 这里默认使用了quadruplet loss，所以必须找到other_neg
        if (q_tuples[3].shape[0] != self.num_points):
            print('----' + 'FAULTY other_neg' + '-----')
            if self.last==[]:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]

        queries = np.expand_dims(np.array(q_tuples[0], dtype=np.float32), axis=0)
        other_neg = np.expand_dims(np.array(q_tuples[3], dtype=np.float32), axis=0)
        positives = np.array(q_tuples[1], dtype=np.float32)
        negatives = np.array(q_tuples[2], dtype=np.float32)

        if (len(queries.shape) != 3):
            print('----' + 'FAULTY QUERY' + '-----')
            if self.last==[]:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]
        self.last = [queries, positives, negatives, other_neg]
        return queries.astype('float32'), positives.astype('float32'), negatives.astype('float32'), other_neg.astype('float32')

    def __len__(self):
        return self.train_len

class Oxford_train_advance(Dataset):
    def __init__(self, args):
        self.num_points = args.num_points
        self.positives_per_query = args.positives_per_query
        self.negatives_per_query = args.negatives_per_query
        self.train_len = len(TRAINING_QUERIES.keys())
        # self.train_file_items = np.random.permutation(np.arange(0, self.train_len))
        # self.train_file_items = np.arange(0, self.train_len)
        print('Load Oxford Dataset')
        # self.data, self.label = []
        self.sampled_neg = 4000
        self.hard_neg_num = args.hard_neg_per_query
        if self.hard_neg_num > args.negatives_per_query:
            print("self.hard_neg_num >  args.negatives_per_query")
        self.last=[]
    def __getitem__(self, item):
        if (len(TRAINING_QUERIES[item]["positives"]) < self.positives_per_query):
            print("lack positive")
            if self.last==[]:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]
        if (len(HARD_NEGATIVES.keys()) == 0):
            query = get_feature_representation(TRAINING_QUERIES[item]['query'], para.model)
            random.shuffle(TRAINING_QUERIES[item]['negatives'])
            negatives = TRAINING_QUERIES[item]['negatives'][0:self.sampled_neg]
            # 找到离当前query最近的neg
            hard_negs = get_random_hard_negatives(query, negatives, self.hard_neg_num)
            # print(hard_negs)
            # q_tuples=get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
            #                                 TRAINING_QUERIES, hard_negs, other_neg=True)
            q_tuples=get_query_tuple_fast(item, TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                TRAINING_QUERIES, hard_neg=hard_negs, other_neg=True)
        #     如果指定了一些HARD_NEGATIVES，實際沒有
        else:
            query = get_feature_representation(
                TRAINING_QUERIES[item]['query'], para.model)
            random.shuffle(TRAINING_QUERIES[item]['negatives'])
            negatives = TRAINING_QUERIES[item
                        ]['negatives'][0:self.sampled_neg]
            hard_negs = get_random_hard_negatives(
                query, negatives, self.hard_neg_num)
            hard_negs = list(set().union(
                HARD_NEGATIVES[item], hard_negs))
            # print('hard', hard_negs)
            # q_tuples=get_query_tuple(TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
            #                     TRAINING_QUERIES, hard_negs, other_neg=True)
            q_tuples=get_query_tuple_fast(item, TRAINING_QUERIES[item], self.positives_per_query, self.negatives_per_query,
                                TRAINING_QUERIES, hard_neg=hard_negs, other_neg=True)

        # 这里默认使用了quadruplet loss，所以必须找到other_neg
        if (q_tuples[3].shape[0] != self.num_points):
            print('----' + 'FAULTY other_neg' + '-----')
            if self.last==[]:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]

        queries = np.expand_dims(np.array(q_tuples[0], dtype=np.float32), axis=0)
        other_neg = np.expand_dims(np.array(q_tuples[3], dtype=np.float32), axis=0)
        positives = np.array(q_tuples[1], dtype=np.float32)
        negatives = np.array(q_tuples[2], dtype=np.float32)

        if (len(queries.shape) != 3):
            print('----' + 'FAULTY QUERY' + '-----')
            if self.last==[]:
                print("wrong")
            else:
                return self.last[0], self.last[1], self.last[2], self.last[3]
        self.last = [queries, positives, negatives, other_neg]
        return queries.astype('float32'), positives.astype('float32'), negatives.astype('float32'), other_neg.astype('float32')

    def __len__(self):
        return self.train_len


def update_vectors(args, model):
    global TRAINING_LATENT_VECTORS
    global TRAINING_QUERIES

    train_file_idxs = np.arange(0, len(TRAINING_QUERIES.keys()))

    batch_num = args.batch_num_queries * \
                (1 + args.positives_per_query + args.negatives_per_query + 1)
    # print("\n args: ",args.batch_num_queries,args.positives_per_query,args.negatives_per_query)
    q_output = []

    model.eval()

    for q_index in tqdm(range(len(train_file_idxs) // batch_num)):
    # for q_index in tqdm(range(batch_num*2 // batch_num)):
        file_indices = train_file_idxs[q_index *
                                       batch_num:(q_index + 1) * (batch_num)]
        file_names = []
        for index in file_indices:
            file_names.append(TRAINING_QUERIES[index]["query"])
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
    if (len(q_output) != 0):
        q_output = q_output.reshape(-1, q_output.shape[-1])

    # handle edge case
    for q_index in tqdm(range((len(train_file_idxs) // batch_num * batch_num), len(TRAINING_QUERIES.keys()))):
        index = train_file_idxs[q_index]
        queries = load_pc_files([TRAINING_QUERIES[index]["query"]])
        queries = np.expand_dims(queries, axis=1)

        with torch.no_grad():
            queries_tensor = torch.from_numpy(queries).float()
            queries_tensor = queries_tensor.to(device)
            o1 = model(queries_tensor)

        output = o1.detach().cpu().numpy()
        output = np.squeeze(output)
        if (q_output.shape[0] != 0):
            q_output = np.vstack((q_output, output.reshape(-1,cfg.FEATURE_OUTPUT_DIM)))
        else:
            q_output = output
    model.train()

    TRAINING_LATENT_VECTORS = q_output
    print("Updated cached feature vectors")