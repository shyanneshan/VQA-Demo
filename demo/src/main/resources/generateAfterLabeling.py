import csv
import random
pos = ["metacarpal", "ulna", "elbow",
     "medial malleolus","radial head","knee","ulnar",
     "lateral malleolus","posterior malleolus",
     "knees","femoral neck","palm",
     "toe","tibial","tibia","talofibular",
     "talus","ankle","fibula","Mid-bone",
     "vertebral","cuboid","femur","wrist",
     "calcaneus","brain","patella",
     "carpal","radioulnar","maxillary",
     "cheekbone","orbital bone","distal radius",
     "lumbar","acetabulum","hip",
     "olecranon","fibular","humerus",
     "meniscus","clavicle","sternal",
     "scaphoid","radius styloid","shoulder",
     "acromioclavicular joint"]
opt1=['axial','coronal','sagittal']
opt2=['A-P','Lateral','Lordotic','A-p supine','P-A']
temp=["Describe the situation in the image? ",
      "What abnormalities are seen within the POS?",
      "Describe the POS abnormalities?",
      "What abnormalities are in the POS?",
      "What is seen in the POS?",
      "What is happening with the POS?"]

def generateQA(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        qid = 0
        for row in reader:
            id = row[0]
            patient_id = row[1]
            photo_id = row[2]
            dia_list = row[3]
            diaList = dia_list.split(',')  # 医生标记出的图片对应的疾病数组（中文表述）
            description = row[4]  # 对片子的描述（中文表述）
            organ = row[5]
            if (organ == '1'):
                image_organ = 'HEAD'  # 图片所属部位
            elif (organ == '2'):
                image_organ = 'CHEST'
            elif (organ == '3'):
                image_organ = 'HAND'
            elif (organ == '4'):
                image_organ = 'LEG'
            else:
                image_organ = ''
            plane = row[6]  # 横断面
            if (plane == '1'):
                image_plane = 'axial'  # 图片拍摄的角度
            elif (plane == '2'):
                image_plane = 'coronal'
            elif (plane == '3'):
                image_plane = 'sagittal'
            else:
                image_plane = ''
            type = row[7]
            if (type == '1'):
                modality = 'CT'  # 片子类型
            elif (type == '2'):
                modality = 'X-Ray'
            else:
                modality = ''
            direction = row[8]
            if (direction == '1'):
                image_dir = 'A-P'  # 图片拍摄的角度
            elif (direction == '2'):
                image_dir = 'Lateral'
            elif (direction == '3'):
                image_dir = 'Lordotic'
            elif (direction == '4'):
                image_dir = 'A-p supine'
            elif (direction == '5'):
                image_dir = 'P-A'
            else:
                image_dir = ''


            q1 = "Is this a ct image or xray image?"
            if(type=='1'):
                a1='CT'
            else:
                a1='X-Ray'
            image_name = photo_id + ".png"
            answer_type = 'CLOSE'
            question_type = 'ModalityType'
            phrase_type = 'fixed'
            Q1 = [qid, image_name, image_organ, a1, answer_type, question_type, q1, phrase_type]
            with open("Modality_CLOSE.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(Q1)
            qid = qid + 1


import os
import argparse

import pymysql
from pymysql.converters import escape_string
from pprint import pprint
import pandas

class Test_myqsl(object):
    #运行数据库和建立游标对象
    def __init__(self):
        self.connect = pymysql.connect(host="127.0.0.1", port=3306, user="root", password="123456", database="vqademo",
                                  charset="utf8")
        # 返回一个cursor对象,也就是游标对象
        self.cursor = self.connect.cursor(cursor=pymysql.cursors.DictCursor)
    #关闭数据库和游标对象
    # def __del__(self):
    #     self.connect.close()
    #     self.cursor.close()
    def write(self,csv_path):
        #将数据转化成DataFrame数据格式
        data = pandas.DataFrame(self.read(csv_path))
        data = data[data['dataset']==csv_path]
        #把id设置成行索引
        data_1 = data.set_index("id",drop=True)
        #写写入数据数据
        pandas.DataFrame.to_csv(data_1,csv_path+".csv",encoding="utf_8_sig")
        print("写入成功")
    def read(self,csv_path):
        #读取数据库的所有数据
        data = self.cursor.execute("""select * from ct_validation;""")
        field_2 = self.cursor.fetchall()
        # pprint(field_2)
        return field_2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='test')
    args = parser.parse_args()
    dataset = args.name
    write = Test_myqsl()
    write.write(dataset)
    generateQA(dataset+'.csv')







