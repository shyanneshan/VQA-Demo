import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

class Utility(object):
    #txt格式：imag_name|question|answer
    @staticmethod
    def clean(line):
        line = line.lower()
        line = line.replace('?', ' ?')
        line = re.sub(' +',' ', line)
        return line.strip()

    @staticmethod
    def read_train_dataset(txt_path,image_dic):
        questions=[]
        answers=[]
        images=[]
        with open(txt_path,'r',encoding='utf-8') as f:
            for line in f:
                line=line.split('\n')[0]
                images.append(os.path.join(image_dic,line.split('|')[0])+'.jpg')
                questions.append(line.split('|')[1])
                answers.append(line.split('|')[2])
        # print("trian data size=",len(images))
        return images, questions,answers

    @staticmethod
    def read_val_dataset(txt_path,image_dic):
        questions = []
        answers = []
        images = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split('\n')[0]
                images.append(os.path.join(image_dic, line.split('|')[0])+'.jpg')
                questions.append(line.split('|')[1])
                answers.append(line.split('|')[2])
        # print("val data size=", len(images))
        return images, questions, answers

    @staticmethod
    def read_test_dataset(txt_path,image_dic):
        questions = []
        answers = []
        images = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split('\n')[0]
                images.append(os.path.join(image_dic, line.split('|')[0])+'.jpg')
                questions.append(line.split('|')[1])
                answers.append(line.split('|')[2])
        # print("test data size=", len(images))
        return images, questions, answers

    @staticmethod
    def read_predict(question,imag):
        questions = [question]
        answers = ["ans"]
        images = [imag]
        # with open(txt_path, 'r', encoding='utf-8') as f:
        #     for line in f:
        #         line = line.split('\n')[0]
        #         images.append(os.path.join(image_dic, line.split('|')[0])+'.jpg')
        #         questions.append(line.split('|')[1])
        #         answers.append(line.split('|')[2])
        # print("test data size=", len(images))
        return images, questions, answers

    @staticmethod
    def show_image(id, images, questions, answers):
        fname = images[id]
        img=mpimg.imread(fname, format="jpg")
        print ("Image name :", fname)
        print ("Question   :", questions[id])
        print ("Answer     :", answers[id] )
        plt.imshow(img)
        plt.show()
