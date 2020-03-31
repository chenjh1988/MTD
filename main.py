# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import tensorflow as tf
import random
from model import MTD
from utils import split_validation_v1, split_validation_v2, Data, Graph
import logging
import pickle
import argparse
import time
import sys
import json
import pdb
import math


def m_print(log):
    logging.info(log)
    print(log)
    return


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_64', help='dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample')
parser.add_argument('--method', type=str, default='sa', help='sa')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size')
parser.add_argument('--emb_size', type=int, default=100, help='hidden state size')
parser.add_argument('--l2', type=float, default=0.000001, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--nonhybrid', action='store_true', help='global preference')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=32, help='the number of steps after which the learning rate decay')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
parser.add_argument('--rec', type=int, default=5, help='the frequence of recommendation loss')

parser.add_argument('--kg', type=int, default=1000, help='the max number of dgi optimizer')
parser.add_argument('--rand_seed', type=int, default=1111)
parser.add_argument('--log_file', type=str, default='./default.txt')
parser.add_argument('--trunc', type=int, default=4, help='stop the dgi')
parser.add_argument('--num_head', type=int, default=1, help='number of self attention multi-head')
parser.add_argument('--num_block', type=int, default=1, help='number of self attention block')
parser.add_argument('--num_gcn', type=int, default=1, help='number of gcn layer')
parser.add_argument('--use_bias', action='store_true', help='use bias')
parser.add_argument('--save_tk', action='store_true', help='save recommendation result')

opt = parser.parse_args()
random.seed(opt.rand_seed)
np.random.seed(opt.rand_seed)
tf.set_random_seed(opt.rand_seed)
logging.basicConfig(level=logging.INFO, format='%(message)s', filename=opt.log_file, filemode='w')

all_train_seq = pickle.load(open('./datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))
if opt.validation:
    if opt.kg > 0:
        train_data, test_data, all_train_seq = split_validation_v2(all_train_seq, frac=0.1)
    else:
        train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
        train_data, test_data = split_validation_v1(train_data, frac=0.1)
else:
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))

# pdb.set_trace()
if opt.dataset == 'diginetica':
    n_node = 43098
elif opt.dataset == 'yoochoose1_64':
    n_node = 37484
elif opt.dataset == 'yoochoose1_4':
    n_node = 37484

train_data = Data(train_data,
                  all_seqs=all_train_seq,
                  sub_graph=True,
                  method=opt.method,
                  shuffle=True)
test_data = Data(test_data,
                 sub_graph=True,
                 method=opt.method,
                 shuffle=False)
graph = Graph(all_train_seq)
biases, node_list = graph.generate_gat()

model = MTD(hidden_size=opt.hidden_size,
            emb_size=opt.emb_size,
            n_node=n_node,
            method=opt.method,
            lr=opt.lr,
            l2=opt.l2,
            decay=opt.lr_dc_step * len(train_data.inputs) / opt.batch_size,
            lr_dc=opt.lr_dc,
            dropout=opt.dropout,
            kg=opt.kg,
            num_head=opt.num_head,
            num_block=opt.num_block,
            nonhybrid=opt.nonhybrid,
            num_gcn=opt.num_gcn)

m_print(json.dumps(opt.__dict__, indent=4))
m_print('train len: %d, test len: %d' % (train_data.length, test_data.length))
best_result = [0, 0]
best_epoch = [0, 0]

for epoch in range(opt.epoch):
    m_print('<==================epoch: %d==================>' % epoch)
    m_print('start train: ' + time.strftime('%m-%d %H:%M:%S ', time.localtime(time.time())))

    if epoch < opt.trunc:
        best = 1e9
        best_h = None
        patience = 10
        for kg_step in range(opt.kg):
            fetches_node = [model.dgi_loss, model.dgi_opt, model.global_step_dgi, model.hh]
            train_start = time.time()
            shuf_list = np.random.permutation(node_list.shape[1])
            shuf_list = node_list[:, shuf_list]
            # pdb.set_trace()
            s_loss, _, _, hh = model.run_dgi(fetches_node, node_list, shuf_list, biases)
            cost_time = time.time() - train_start
            if kg_step % 100 == 0:
                m_print('Step: %d, Train kg_Loss: %.4f, Cost: %.2f' % (kg_step, s_loss, cost_time))
            if s_loss < best:
                best = s_loss
                best_h = hh
                cnt_wait = 0
            else:
                cnt_wait += 1
            if cnt_wait == patience:
                m_print('Step: %d, Train kg_Loss: %.4f, Cost: %.2f' % (kg_step, s_loss, cost_time))
                print('Early stopping!')
                break
        embs = model.sess.run(model.embedding)
        embs[node_list.flatten(), :] = best_h
        embs = tf.convert_to_tensor(embs)
        model.sess.run(tf.assign(model.pre_embedding, embs))

    for rec_step in range(opt.rec):
        slices = train_data.generate_batch(opt.batch_size)
        fetches = [model.rec_opt, model.rec_loss, model.global_step]
        loss_ = []
        for i, j in zip(slices, np.arange(len(slices))):
            batch_input = train_data.get_slice(i)
            _, loss, _ = model.run_rec(fetches, batch_input)
            loss_.append(loss)
        loss = np.mean(loss_)

        slices = test_data.generate_batch(opt.batch_size)
        m_print('start predict:' + time.strftime('%m-%d %H:%M:%S ', time.localtime(time.time())))
        hit = {5: [], 10: [], 15: [], 20: [], 30: [], 40: [], 50: [], 60: []}
        mrr = {5: [], 10: [], 15: [], 20: [], 30: [], 40: [], 50: [], 60: []}
        ndcg = {5: [], 10: [], 15: [], 20: [], 30: [], 40: [], 50: [], 60: []}
        test_loss_, ans = [], []
        for i, j in zip(slices, np.arange(len(slices))):
            batch_input = test_data.get_slice(i)
            scores, test_loss, tk = model.run_rec(
                [model.logits, model.rec_loss, model.top_k], batch_input, is_train=False)
            test_loss_.append(test_loss)
            ans.append(tk)

            targets = batch_input[-1]
            for score, target in zip(tk, targets):
                for i in [5, 10, 15, 20, 30, 40, 50, 60]:
                    # pdb.set_trace()
                    hit[i].append(np.isin(target - 1, score[:i]))
                    if len(np.where(score[:i] == target - 1)[0]) == 0:
                        mrr[i].append(0)
                        ndcg[i].append(0)
                    else:
                        rank = 1 + np.where(score[:i] == target - 1)[0][0]
                        mrr[i].append(1 / rank)
                        ndcg[i].append(1 / math.log(rank + 1, 2))
        for i in [5, 10, 15, 20, 30, 40, 50, 60]:
            hit[i] = np.mean(hit[i]) * 100
            mrr[i] = np.mean(mrr[i]) * 100
            ndcg[i] = np.mean(ndcg[i]) * 100

        test_loss = np.mean(test_loss_)
        if hit[20] >= best_result[0]:
            best_result[0] = hit[20]
            best_epoch[0] = epoch
            if opt.save_tk:
                pickle.dump(ans, open(opt.log_file.replace('txt', 'hit'), 'wb'))
        if mrr[20] >= best_result[1]:
            best_result[1] = mrr[20]
            best_epoch[1] = epoch
            if opt.save_tk:
                pickle.dump(ans, open(opt.log_file.replace('txt', 'mrr'), 'wb'))
        m_print('train_loss: %.4f, test_loss: %.4f' % (loss, test_loss))
        m_print('Recall@5: %.4f, MMR@5: %.4f, NDCG@5: %.4f' % (hit[5], mrr[5], ndcg[5]))
        m_print('Recall@10: %.4f, MMR@10: %.4f, NDCG@10: %.4f' % (hit[10], mrr[10], ndcg[10]))
        m_print('Recall@15: %.4f, MMR@15: %.4f, NDCG@15: %.4f' % (hit[15], mrr[15], ndcg[15]))
        m_print('Recall@20: %.4f, MMR@20: %.4f, NDCG@20: %.4f' % (hit[20], mrr[20], ndcg[20]))
        m_print('Recall@30: %.4f, MMR@30: %.4f, NDCG@30: %.4f' % (hit[30], mrr[30], ndcg[30]))
        m_print('Recall@40: %.4f, MMR@40: %.4f, NDCG@40: %.4f' % (hit[40], mrr[40], ndcg[40]))
        m_print('Recall@50: %.4f, MMR@50: %.4f, NDCG@50: %.4f' % (hit[50], mrr[50], ndcg[50]))
        m_print('Recall@60: %.4f, MMR@60: %.4f, NDCG@60: %.4f' % (hit[60], mrr[60], ndcg[60]))
        m_print('Best Recall@20: %.4f, Best MMR@20: %.4f, Epoch: %d, %d\n' %
                (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
