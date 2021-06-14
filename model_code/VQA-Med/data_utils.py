import re
import pandas as pd
import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


class Utility(object):
    @staticmethod
    def clean(line):
        line = line.lower()
        line = line.replace('?', ' ?')
        line = re.sub(' +',' ', line)
        return line.strip()

    @staticmethod
    def read_dataset(df,mode):

        if "Test" in mode:
            df = df.rename(columns={0: 'id', 1: 'image_name', 2:"question"})
        else:
            df = df.rename(columns={0: 'id', 1: 'image_name', 2:"question", 3:"answer"})
        # print(dataset+" data size=",len(df))
        images = []

        for i in df['image_name']:
            # fname = "/home/wxl/Documents/model/VQA-Med/dataset/VQAMed2018"+dataset+"/VQAMed2018"+dataset+"-images/"+i+".jpg"
            images.append(i)
        if "Test" in mode:
            return images, df["question"]
        else:
            return images, df["question"], df["answer"]


    @staticmethod
    def show_image(id, images, questions, answers):
        fname = images[id]
        img=mpimg.imread(fname, format="jpg")
        # print ("Image name :", fname)
        # print ("Question   :", questions[id])
        # print ("Answer     :", answers[id] )
        plt.imshow(img)
        plt.show()
