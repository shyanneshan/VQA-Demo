# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:57:57 2020
@author: Xuandi Fu(Carnegie Mellon University)
"""
import argparse
import os
import json
import numpy as np
# from language.glove import Glove
import nltk
from string import punctuation
import numpy
from tqdm import trange
import sys

glove_path = 'glove/glove.6B.50d.txt'
train_path = 'data/okvqa_an_input_train.txt'
val_path = 'data/okvqa_an_input_val.txt'
# glove_obj = Glove(glove_path)
embeddings_dict = {}

with open("/data2/entity/bhy/VQADEMO/model_code/ArticleNet/glove/glove.6B.50d.txt", 'r', encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector
import torch

# glove_obj = torch.Tensor(np.load('../data/wordVectors.npy')).float()
# print(glove_obj.dim)




def __process_sent(sent,translate_table):
    return sent.lower().translate(translate_table)


def get_sent_embedding(sentence,translate_table, keep_pos=None):
    sentence = __process_sent(sentence,translate_table)
    word_tokens = nltk.word_tokenize(sentence)
    tokens = nltk.pos_tag(word_tokens)
    tokens = [item[0] for item in tokens if keep_pos is None or item[1] in keep_pos]
    dim = 50
    sent_emb = np.zeros(dim)
    cnt = 0
    for word in tokens:
        # emb = glove_obj.embedding.get(word, None)
        try:
            emb = embeddings_dict[word]
        except:
            emb = None

        if emb is not None:
            sent_emb += emb
    return sent_emb / cnt if cnt > 0 else sent_emb


def processImag(mode, datasetpath,imags):
    import cv2
    import numpy as np
    import os
    data_path = os.path.join(datasetpath, mode + "_an.json")
    with open(data_path, 'r') as f:
        data = json.load(f)
        for qa in data:
            image_path = qa['image_name']
            img = cv2.imread(os.path.join(datasetpath, mode,image_path))
            img.resize((3, 128, 128))
            imags.append(img)
    np.save(os.path.join(datasetpath, mode+"_npy","imag.npy"), imags)


def loadData(path, question_emb,
             title_emb, sentence_emb, labels,translate_table):
    train_file = open(path, 'r')
    line_num = 0
    pos_label = 0
    lines = train_file.readlines()
    length = []
    for i in trange(len(lines), file=sys.stdout, desc='outerloop'):
        train_data = json.loads(lines[i])
        line_num += 1
        # ===================load embedding==================
        question_emb.append(get_sent_embedding(train_data['question'],translate_table))
        title_emb.append(get_sent_embedding(train_data['title'],translate_table))

        sentence = []
        sentence_label = []
        length.append(len(train_data['sentences_pairs']))
        for pair in train_data['sentences_pairs']:
            sentence.append(get_sent_embedding(pair[0],translate_table))
            sentence_label.append(pair[1])
        sentence_emb.append(sentence)
        # labels.append(sentence_label)

        # =================load label======================
        labels.append([train_data['titleHasAns']] + sentence_label + [train_data['docHasAns']])
        # print(get_sent_embedding(train_data['question']))


def saveData(npypath, question_emb, sentence_emb):
    # global question_emb, sentence_emb

    question_emb = numpy.array(question_emb)
    sentence_emb = numpy.array(sentence_emb)

    with open(os.path.join(npypath, 'question.npy'), 'wb') as f:
        np.save(f, question_emb)
    with open(os.path.join(npypath, 'sentence.npy'), 'wb') as f:
        np.save(f, sentence_emb)


def saveLabel(npypath, labels, title_emb):
    # global labels
    # global title_emb
    labels = numpy.array(labels)
    title_emb = numpy.array(title_emb)

    with open(os.path.join(npypath, 'label.npy'), 'wb') as f:
        np.save(f, labels)
    with open(os.path.join(npypath, 'title.npy'), 'wb') as f:
        np.save(f, title_emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="dataset", help='dataset origin folder.')

    args = parser.parse_args()
    root = args.dataset
    modes = ['train', 'val', 'test']
    translate_table = dict((ord(char), None) for char in punctuation)
    for mode in modes:
        if not os.path.exists(os.path.join(root,mode+"_npy")):
            os.mkdir(os.path.join(os.path.join(root,mode+"_npy")))
        question_emb = []
        title_emb = []
        sentence_emb = []
        imags = []
        labels = []
        loadData(os.path.join(root, "input_" + mode + '.txt'), question_emb,
                 title_emb, sentence_emb, labels)
        saveData(os.path.join(root, mode + "_npy"), question_emb, sentence_emb)
        saveLabel(os.path.join(root, mode + "_npy"), labels, title_emb)
        processImag(mode, root,imags)
