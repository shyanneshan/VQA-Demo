import csv

import pandas as pd
import os

def getTrain_ValCsv(path):
    for file in os.listdir(path):
        filepath=os.path.join(path,file)
        category=file.split('.txt')[0].split('_')[1]
        Data={'img_id':[],'question':[],'answer':[],'category':[]}
        with open(filepath, 'r', encoding='utf-8') as f:
            linecount = 1  # 从第一行开始计数id
            for line in f:
                content = line.split('|')
                Data['category'].append(category)
                Data['img_id'].append(content[0])
                Data['question'].append(content[1])
                Data['answer'].append(content[2].split('\n')[0])
                linecount += 1
    return Data

def getTestCsv(args):
    df = pd.read_csv(args.test_text_file, sep='|', names=['img_id', 'question', 'answer'])
    df['mode'] = "test"
    df['img_id'] = df['img_id'].apply(lambda x: os.path.join(args.test_image_file, x + '.jpg'))
    return df

def getTrainCsv( args):
    df = pd.read_csv(args.train_text_file, sep='|', names=['img_id', 'question', 'answer'])
    df['mode'] = "train"
    df['img_id'] = df['img_id'].apply(lambda x: os.path.join(args.train_image_file, x + '.jpg'))
    return df

def getValidCsv( args):
    df = pd.read_csv(args.valid_text_file, sep='|', names=['img_id', 'question', 'answer'])
    df['mode'] = "val"
    df['img_id'] = df['img_id'].apply(lambda x: os.path.join(args.valid_image_file, x + '.jpg'))
    return df
    # testData = {'img_id': [], 'question': [], 'answer': [], 'category': [],'mode':[]}
    # with open(path, 'r', encoding='utf-8') as f:
    #     linecount = 1  # 从第一行开始计数id
    #     for line in f:
    #         content = line.split('|')
    #         testData['mode'].append('test')
    #         testData['img_id'].append(content[0])
    #         testData['category'].append(content[1])
    #         testData['question'].append(content[2])
    #         testData['answer'].append(content[3].split('\n')[0])
    #         linecount += 1
    # return testData

# def change_fromAll_toTxt(file):
#     category=['Modality','Plane','Organ','Abnormality']
#     testData = {'img_id': [], 'question': [], 'answer': [], 'category': []}
#     with open(file,'r',encoding='utf-8') as f:
#         for line in f:
#             content = line.split('|')
#             testData['img_id'].append(content[0])
#             testData['category'].append(content[1])
#             testData['question'].append(content[2])
#             testData['answer'].append(content[3].split('\n')[0])
#     Modality={}
#     Plane = {}
#     Organ = {}
#     Abnormality = {}
#     for item in testData:
#         if item['category'] ==category[0]:
#             Modality+=item
#         if item['category'] ==category[1]:
#             Plane+=item
#         if item['category'] ==category[2]:
#             Organ+=item
#         if item['category'] ==category[3]:
#             Abnormality+=item
#     for i in range(category):
#         with open(os.path.join(path,category[i]),'w',encoding='utf-8') as fw:
#


if __name__=='__main__':
    trainData=getTrain_ValCsv(r'C:\Users\PC\Documents\GitHub\VQA-Med-2019\ImageClef-2019-VQA-Med-Training\QAPairsByCategory')
    testData=getTestCsv(r'C:\Users\PC\Documents\GitHub\VQA-Med-2019\VQAMed2019Test\VQAMed2019_Test_Questions_w_Ref_Answers.txt')
    valData=getTrain_ValCsv(r'C:\Users\PC\Documents\GitHub\VQA-Med-2019\ImageClef-2019-VQA-Med-Validation\QAPairsByCategory')

    df_train=pd.DataFrame(trainData)
    df_train.to_csv('dataset\\traindf.csv',index=False)

    df_test=pd.DataFrame(testData)
    df_test.to_csv('dataset\\testdf.csv',index=False)

    df_val=pd.DataFrame(valData)
    df_val.to_csv('dataset\\valdf.csv',index=False)