# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:57:57 2020
@author: Xuandi Fu(Carnegie Mellon University)
""
"""
import argparse
import configparser

from sklearn import *
from model import ArticleNet
from torch.utils.data import DataLoader, TensorDataset
import torch
from dataLoader import get_dataloader
import os

# config = configparser.ConfigParser()
# config.read('config.ini')
model_root = 'data/save'


#
# def get_config():
#     config = {}
#     config['num_epoch'] = 20
#     config['batch_size'] = 64
#     config['lr'] = 0.001
#     config['q_len']= 50
#     config['q_hidden'] = 128
#     config['q_layer'] = 1#1
#     config['tit_len'] = 50
#     config['tit_hidden'] = 128
#     config['t_layer'] = 1#1
#     config['sent_len'] = 50
#     config['sent_hidden'] = 128
#     config['s_layer'] = 1#1
#     config['art_hidden'] = 128
#     config['title_num_class'] = 1#1
#     config['sent_num_class'] = 1100
#     config['art_num_class'] = 1#1
#
#     config['val_size'] = 761
#
#     return config

def save_checkpoint(model_dict, model_path):
    torch.save(model_dict, model_path)
    return


def cointinuous2binary(l):
    for i in range(len(l)):
        max_value = max(l[i])
        for j in range(len(l[i])):
            if l[i][j] == max_value:
                l[i][j] = 1
            else:
                l[i][j] = 0
    return l


def train(args):
    model = ArticleNet(args).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bceloss = torch.nn.BCELoss()
    train_loader = get_dataloader("dataset", 'train', shuffle=True, batch_size=args.batch_size)
    # val_loader = get_dataloader('val', shuffle=False, batch_size=config['val_size'])
    val_loader = get_dataloader("dataset", 'val', shuffle=False, batch_size=args.batch_size)
    test_loader = get_dataloader("dataset", 'test', shuffle=False, batch_size=args.batch_size)

    for epoch in range(args.num_epoch):
        # =========================== train =====================
        model.train()
        with open('artpredict.txt', 'a+') as file_handle:
            file_handle.write('epoch:' + str(epoch) + '\n')
        print("epoch: ", str(epoch))
        for qbatch, vbatch, tbatch, sbatch, ybatch, len_batch in train_loader:
            ypred = model.forward(qbatch, vbatch, tbatch, sbatch, len_batch, args)
            optimizer.zero_grad()

            ypred = torch.hstack((ypred[0], ypred[1], ypred[2]))
            loss = bceloss(ypred, ybatch)


            loss.backward()
            optimizer.step()
        # =========================== eval =====================
        print("eval..............")
        import numpy as np
        model.eval()
        total_a_title = torch.tensor([])
        total_a_sent = torch.tensor([])
        total_a_art = torch.tensor([])
        total_y_title = torch.tensor([])
        total_y_sent = torch.tensor([])
        total_y_art = torch.tensor([])

        sent_right_num = 0
        sent_num = 0
        i_count = 0
        with torch.no_grad():
            for qbatch, vbatch, tbatch, sbatch, ybatch, len_batch in val_loader:

                a_title, a_sent, a_art = model.forward(qbatch, vbatch, tbatch, sbatch, len_batch, args)

                # 分别计算a_title, a_sent, a_art
                y_title = ybatch[:, 0]
                y_sent = ybatch[:, 1:-1]
                y_art = ybatch[:, -1]

                # a_sent = np.rint(a_sent)
                a_sent = cointinuous2binary(a_sent)
                a_title = cointinuous2binary(a_title)
                a_art = cointinuous2binary(a_art)

                try:
                    total_a_art = torch.hstack((total_a_art, a_art))
                    total_a_sent = torch.hstack((total_a_sent, a_sent))
                    total_a_title = torch.hstack((total_a_title, a_title))
                    total_y_art = torch.hstack((total_y_art, y_art))
                    total_y_sent = torch.hstack((total_y_sent, y_sent))
                    total_y_title = torch.hstack((total_y_title, y_title))
                except:
                    pass
                use_count = 0

                for i in range(len(y_sent)):
                    if (torch.equal(y_sent[i], a_sent[i])):
                        single_score = 1
                    else:
                        single_score = 0
                    use_count = use_count + single_score
                    i_count = i_count + 1
                sent_score = use_count / 64


                sent_right_num = sent_score * len(a_sent) + sent_right_num
                sent_num = sent_num + len(a_sent)

        # =========================== save model =====================
        # print("len_title:",title_num)
        sent_result = float(sent_right_num) / sent_num
        print("sent_result:", sent_result)
        save_checkpoint(model.state_dict(), args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', default="save/dataset.pt", help='model save path.')
    parser.add_argument('--num_epoch', default=20, help='epoch number')
    parser.add_argument('--batch_size', default=64)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--q_len', default=50)
    parser.add_argument('--q_hidden', default=128)
    parser.add_argument('--q_layer', default=1)
    parser.add_argument('--tit_len', default=50)
    parser.add_argument('--tit_hidden', default=128)
    parser.add_argument('--t_layer', default=1)
    parser.add_argument('--sent_len', default=50)
    parser.add_argument('--sent_hidden', default=128)
    parser.add_argument('--s_layer', default=1)
    parser.add_argument('--art_hidden', default=128)
    parser.add_argument('--title_num_class', default=1)
    parser.add_argument('--sent_num_class', default=1100)
    parser.add_argument('--art_num_class', default=1)

    args = parser.parse_args()

    model = ArticleNet(args).double()

    model_dict = train(args)