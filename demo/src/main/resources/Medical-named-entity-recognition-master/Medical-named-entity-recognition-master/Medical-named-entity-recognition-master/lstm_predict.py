import json

import numpy as np
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras_contrib.layers.crf import CRF
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class LSTMNER:
    def __init__(self):
        cur = '/'.join(os.path.abspath(__file__).split('/')[:-1])
        self.train_path = os.path.join(cur, 'data/train.txt')
        self.vocab_path = os.path.join(cur, 'model/vocab.txt')
        self.embedding_file = os.path.join(cur, 'model/token_vec_300.bin')
        self.model_path = os.path.join(cur, 'model/tokenvec_bilstm2_crf_model_20.h5')
        self.word_dict = self.load_worddict()
        self.class_dict ={
                         'O':0,
                         'TREATMENT-I': 1,
                         'TREATMENT-B': 2,
                         'BODY-B': 3,
                         'BODY-I': 4,
                         'SIGNS-I': 5,
                         'SIGNS-B': 6,
                         'CHECK-B': 7,
                         'CHECK-I': 8,
                         'DISEASE-I': 9,
                         'DISEASE-B': 10
                        }
        self.label_dict = {j:i for i,j in self.class_dict.items()}
        self.EMBEDDING_DIM = 300
        self.EPOCHS = 10
        self.BATCH_SIZE = 128
        self.NUM_CLASSES = len(self.class_dict)
        self.VOCAB_SIZE = len(self.word_dict)
        self.TIME_STAMPS = 150
        self.embedding_matrix = self.build_embedding_matrix()
        self.model = self.tokenvec_bilstm2_crf_model()
        self.model.load_weights(self.model_path)

    '加载词表'
    def load_worddict(self):
        with open (self.vocab_path,'r',encoding='utf-8') as f:
            lines=f.read()
        vocabs = lines.split('\n')
        word_dict = {wd: index for index, wd in enumerate(vocabs)}
        return word_dict

    '''构造输入，转换成所需形式'''
    def build_input(self, text):
        x = []
        for char in text:
            if char not in self.word_dict:
                char = 'UNK'
            x.append(self.word_dict.get(char))
        x = pad_sequences([x], self.TIME_STAMPS)
        return x

    def predict(self, text):
        str = self.build_input(text)
        raw = self.model.predict(str)[0][-self.TIME_STAMPS:]
        result = [np.argmax(row) for row in raw]
        chars = [i for i in text]
        tags = [self.label_dict[i] for i in result][len(result)-len(text):]
        res = list(zip(chars, tags))
        # print(res)
        exp2(res)
        return exp2(res)

    '''加载预训练词向量'''
    def load_pretrained_embedding(self):
        embeddings_dict = {}
        with open(self.embedding_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split(' ')
                if len(values) < 300:
                    continue
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_dict[word] = coefs
        # print('Found %s word vectors.' % len(embeddings_dict))
        return embeddings_dict

    '''加载词向量矩阵'''
    def build_embedding_matrix(self):
        embedding_dict = self.load_pretrained_embedding()
        embedding_matrix = np.zeros((self.VOCAB_SIZE + 1, self.EMBEDDING_DIM))
        for word, i in self.word_dict.items():
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                try:
                    embedding_matrix[i] = embedding_vector
                except:
                    print("sth is wrong")

        return embedding_matrix

    '''使用预训练向量进行模型训练'''
    def tokenvec_bilstm2_crf_model(self):
        model = Sequential()
        embedding_layer = Embedding(self.VOCAB_SIZE + 1,
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.TIME_STAMPS,
                                    trainable=False,
                                    mask_zero=True)
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(self.NUM_CLASSES)))
        crf_layer = CRF(self.NUM_CLASSES, sparse_target=True)
        model.add(crf_layer)
        model.compile('adam', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])
        model.summary()
        return model


def concatRes(testdata):
    beginList = ['B-TREATMENT', 'B-BODY', 'B-SIGNS', 'B-CHECK', 'B-DISEASE']
    testdata.strip("\n")
    new_list = []
    origincal_data=testdata.split("\n")
    for i in range(len(origincal_data)):
        if origincal_data[i] == '':
            continue
        data = origincal_data[i]
        if data.split(' ')[1] in beginList:
            entity_type = data.split(' ')[1].split("-")[1]

            left = i
            right = i
            currentRes = data[0]
            for j in range(i + 1, len(origincal_data)):
                if origincal_data[j]=='':
                    break
                currentType = origincal_data[j].split(' ')[1]
                if currentType != 'I-' + entity_type:
                    right = j + 1
                    break
                else:
                    currentRes = currentRes + origincal_data[j][0]
            if left != right:
                tmp = [currentRes, entity_type]
                new_list.append(tmp)

            i = j
    return new_list


def predict(string):
    ner = LSTMNER()
    s = string.strip()

    return concatRes(ner.predict(s))


def predictAll(dicname,destname):
        result=[]
        with open(dicname, 'r', encoding='utf-8') as f: # 要去掉隐私的文件
            for line in f.readlines():
                if len(line)>20:
                    line_split=line.split('，')
                    for l in line_split:
                        res = predict(l)
                        result = result + (res)
                else:
                    res=predict(line)
                    result=result+(res)
        dict={'CHECK':[],
              'SIGNS':[],
              'DISEASE':[],
              'TREATMENT':[],
              'BODY':[]
              }
        for r in result:
            dict[r[1]].append(r[0])
        print(dict)

        with open(destname, "w", encoding='utf-8') as f:
            # indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
            f.write(json.dumps(dict, indent=4))
        return dict['DISEASE']


def exp2(list):
    res=''
    for s in list:
        type='O'
        if s[1] == 'DISEASE-B':
            type = 'B-DISEASE'
        elif s[1] == 'BODY-B':
            type = 'B-BODY'
        elif s[1] == 'SIGNS-B':
            type = 'B-SIGNS'
        elif s[1] == 'CHECK-B':
            type = 'B-CHECK'
        elif s[1] == 'TREATMENT-B':
            type = 'B-TREATMENT'
        elif s[1] == 'DISEASE-I':
            type= 'I-DISEASE'
        elif s[1] == 'BODY-I':
            type = 'I-BODY'
        elif s[1] == 'SIGNS-I':
            type = 'I-SIGNS'
        elif s[1] == 'CHECK-I':
            type = 'I-CHECK'
        elif s[1] == 'TREATMENT-I':
            type= 'I-TREATMENT'
        res=res+s[0]+' '+type+"\n"
    print(res)
    return res
#
# if __name__ == '__main__':
#     ner = LSTMNER()
#     res=''
#     with open('train.txt', 'r') as f2:
#         lines=f2.readlines()
#         for line in lines:
#             res=res+ner.predict(line[:-1])
#     print(res)
#     res.strip()
#     with open('test.char.bmes', 'w', encoding='utf-8') as fw1:
#         fw1.write(res)
        # fw1.write('\n')
    # while 1:
    #
    #     s = input('enter an sent:').strip()
    #
    #     ner.predict(s)
    #     concatRes(ner.predict(s))