import argparse
import json
import _pickle as cPickle
import os
import pickle
import re
import shutil
from PIL import Image #导入PIL库
import torch
import torchvision.transforms as transforms
import cv2


def create_question_json(dataroot):
    # imgid2val = {}
    # print("Start creating set.json files.")
    paths = ['train', 'val', 'test']
    for path in paths:
        path = dataroot + '/' + path
        questions = []
        answers = []
        qid = 0
        with open(path + '.txt', 'r', encoding='utf-8') as f:
            for line in f:
                s = line.split('|')
                img_id = s[0]
                question = s[1]
                answer = s[2]
                # img = cv2.imread('path/' + img_id)
                if answer[-1] == '\n':
                    answer = answer[:-1]
                qid += 1
                # imgid2val[img_id] = img
                questions.append({'qid': qid, 'question': question, 'answer': answer, 'image_name': img_id})
                # answers.append({'qid': qid, 'answer': answer, 'image_name': img_id})
        # print(questions)
        with open(path + 'set.json', 'w', encoding='utf-8') as f:
            json.dump(questions, f, indent=4)
    # print("Finishing creating set.json files.")

# def create_answer_json(dataroot):
#     # imgid2val = {}
#     import pickle
#     ans2label = pickle.load(open('../'+dataroot+'/cache/trainval_ans2label.pkl', 'rb'))
#     print("Start creating *_target.pkl files.")
#     paths = ['train', 'val', 'test']
#     for path in paths:
#         filepath = '../' + dataroot + '/' + path
#         questions = []
#         answers = []
#         qid = 0
#         with open(filepath + '.txt', 'r', encoding='utf-8') as f:
#             for line in f:
#                 s = line.split('|')
#                 img_id = s[0]
#                 question = s[1]
#                 answer = s[2]
#                 # img = cv2.imread('path/' + img_id)
#                 if answer[-1] == '\n':
#                     answer = answer[:-1]
#                 answer=preprocess_answer(answer)
#                 qid += 1
#                 # imgid2val[img_id] = img
#                 # questions.append({'qid': qid, 'question': question, 'answer': answer, 'image_name': img_id})
#                 answers.append({'qid': qid,  'image_name': img_id,
#                                 'labels': [ans2label[answer]],'scores':[1.0]})
#         # print(questions)
#         with open(filepath + '_target.pkl', 'wb') as f:
#             pickle.dump(answers, f)
#     print("Finishing creating *_target.pkl files.")


def mergeimgs(dataroot):
    datapath = dataroot + '/images'
    if not os.path.exists(datapath):
        os.mkdir(datapath)
    # print("create images folder success!")
    paths = ['train', 'val', 'test']
    for path in paths:
        imagespath = dataroot + '/' + path
        for imagename in os.listdir(imagespath):
            imagepath = os.path.join(imagespath, imagename)
            targetpath = os.path.join(datapath, imagename)
            # print(imagepath)
            # print(targetpath)
            shutil.copyfile(imagepath, targetpath)


def create_img2val(dataroot):
    imgid2val = {}
    # print("Start creating img2val.json files.")
    path = dataroot +'/images'
    i = 0
    for imag in os.listdir(path):
        # img_path=os.path.join(path,imag)
        imgid2val[imag.split('.')[0]]=i
        i+=1
    with open(dataroot+'/imgid2idx.json', 'w', encoding='utf-8') as f:
        json.dump(imgid2val, f, indent=4)
    # print("Finish creating img2val.json files.")

def create_84(dataroot):
    # print("Start creating 84 pickle files.")
    res=[]
    folder=dataroot+'/images'
    destpath=dataroot+'/images84x84.pkl'
    for image in os.listdir(folder):
        img_path=os.path.join(folder,image)
        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        img=img.resize((84,84),Image.ANTIALIAS)
        img = img.convert('L')
        transf = transforms.ToTensor()
        img_tensor = transf(img)
        res.append(img_tensor.numpy())
    res=torch.Tensor(res)
    with open(destpath, 'wb') as f:
        pickle.dump(res, f)
    # print("Finish creating 84 pickle files.")

def create_128(dataroot):
    # print("Start creating 128 pickle files.")
    res = []
    folder = dataroot + '/images'
    destpath = dataroot + '/images128x128.pkl'
    for image in os.listdir(folder):
        img_path = os.path.join(folder, image)
        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = img.convert('L')
        transf = transforms.ToTensor()
        img_tensor = transf(img)
        res.append(img_tensor.numpy())
    res = torch.Tensor(res)
    with open(destpath, 'wb') as f:
        pickle.dump(res, f)
    # print("Finish creating 128 pickle files.")


if __name__ == "__main__":
    # 首先生成set.json文件
    # 到create_dictionary.py中生成dictionary
    # 到create_embedding生成embedding文件
    # 到compute_softscore生成target文件
    # 到process中生成 img2val文件
    # 继续生成 84*84 128*128的图片文件
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data_RAD',
                        help='input file directory for continue training from stop one')
    args = parser.parse_args()

    datasetpath = args.dataset

    create_question_json(datasetpath)

    mergeimgs(datasetpath)

    create_img2val(datasetpath)

    create_128(datasetpath)
    create_84(datasetpath)

