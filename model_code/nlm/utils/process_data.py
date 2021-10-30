import json

train_dic = r'D:\program\vqamed2019\VQA-Med-2019\ImageClef-2019-VQA-Med-Training'
val_dic = r'D:\program\vqamed2019\VQA-Med-2019\ImageClef-2019-VQA-Med-Validation'
test_dic = r'D:\program\vqamed2019\VQA-Med-2019\VQAMed2019Test'

train_img = 'Train_images'
val_img = 'Val_images'
test_img = 'VQAMed2019_Test_Images'

train_txt = 'All_QA_Pairs_train.txt'
val_txt = 'All_QA_Pairs_val.txt'
test_txt = 'VQAMed2019_Test_Questions_w_Ref_Answers.txt'

train_json = 'train.json'
val_json = 'val.json'
test_json = 'test.json'

import os

with open(os.path.join(train_dic, train_txt), 'r', encoding='utf-8') as ftrain:
    id = 0
    img_ids = []
    questions = []
    answers = []
    question_ids = []
    train = []
    for line in ftrain:
        img_id = str(line.split('|')[0])
        question = str(line.split('|')[1])
        answer = str(line.split('|')[2])
        question_id = str(id + 1)
        img_ids.append(img_id)
        questions.append(question)
        answers.append(answer)
        question_ids.append(question_id)
        train.append({'question': question, 'question_id': question_id,
                    'answer': answer, 'image_id': img_id})
        id += 1
with open(os.path.join(train_dic,train_json),"w",encoding='utf-8') as f:
    json.dump(train,f)
    # print("加载入Train文件完成...")

with open(os.path.join(val_dic, val_txt), 'r', encoding='utf-8') as fval:
    id = 0
    img_ids = []
    questions = []
    answers = []
    question_ids = []
    val=[]
    for line in fval:
        img_id = str(line.split('|')[0])
        question = str(line.split('|')[1])
        answer = str(line.split('|')[2])
        question_id = str(id + 1)
        img_ids.append(img_id)
        questions.append(question)
        answers.append(answer)
        question_ids.append(question_id)
        val.append({'question':question,'question_id':question_id,
               'answer':answer,'image_id':img_id})
        id += 1
with open(os.path.join(val_dic,val_json),"w",encoding='utf-8') as f:
    json.dump(val,f)
    # print("加载入Val文件完成...")

with open(os.path.join(test_dic, test_txt), 'r', encoding='utf-8') as ftest:
    id = 0
    img_ids = []
    questions = []
    answers = []
    question_ids = []
    test=[]
    for line in ftest:
        img_id = str(line.split('|')[0])
        question = str(line.split('|')[2])
        answer = str(line.split('|')[3])
        question_id = str(id + 1)
        img_ids.append(img_id)
        questions.append(question)
        answers.append(answer)
        question_ids.append(question_id)
        test.append({'question': question, 'question_id': question_id,
                    'answer': answer, 'image_id': img_id})
        id += 1
with open(os.path.join(test_dic,test_json),"w",encoding='utf-8') as f:
    json.dump(test,f)
    # print("加载入Test文件完成...")