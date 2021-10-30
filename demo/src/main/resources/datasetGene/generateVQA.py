import argparse
import csv
import os
import shutil

import mysqlConnector
from ner import ner

''' 
input: filepath
1. insert into ct_information
2. label
3. extract from ct_validation
4. generate VQA

'''


# form picid|Q|A   train val test  the propotion of it must be
def insertInformation(src, dataset_name, dest, train, valid, test):
    for dic in os.listdir(src):
        dicname = src + '/' + dic
        for i in os.listdir(dicname):
            # os.path.splitext():分离文件名与扩展名
            if os.path.splitext(i)[1] == '.jpg':
                imgname = dicname + '/' + i
                imgid = i.split('.')[0]
            elif os.path.splitext(i)[1] == '.txt':
                textname = dicname + '/' + i
                textid = i.split('.')[0]
        print(os.listdir(dicname))

        with open(textname, 'r') as f:
            data = f.read()
            entities = ner(data)
            dialist=''
            entDic={'DISEASE':[],'CHEMICAL':[]}
            for entity in entities:
                if entity[1]=='DISEASE':
                    dialist+=entity[0]+','

            patient_id = textid
            photo_id = imgid
            annotation = ''
            dataset = dataset_name
            mysqlConnector.insert_ct_info(patient_id, data, photo_id, dialist, annotation, dataset)


    return 0


def generateVQA(csv_path, dest, train, valid, test):
    result = []
    trainRes = ''
    validRes = ''
    testRes = ''
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
            image_name = photo_id
            answer_type = 'CLOSE'
            question_type = 'ModalityType'
            phrase_type = 'fixed'
            Q1 = [qid, image_name, image_organ, a1, answer_type, question_type, q1, phrase_type]
            with open("Modality_CLOSE.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(Q1)
            qid = qid + 1
            result.append(image_name + '|' + q1+ '|' + a1)

    length = len(result)
    trainImgList = []
    validImgList = []
    testImgList = []
    trainSampleNum = int(train * length / 10)
    validSampleNum = int(valid * length / 10)
    testSampleNum = length - trainSampleNum - validSampleNum
    for i in range(trainSampleNum):
        trainRes = trainRes + result[i] + '\n'
        trainImgList.append(result[i].split('|')[0])
    for i in range(trainSampleNum, testSampleNum + validSampleNum):
        validRes = validRes + result[i] + '\n'
        validImgList.append(result[i].split('|')[0])
    for i in range(testSampleNum + validSampleNum, length):
        testRes = testRes + result[i] + '\n'
        testImgList.append(result[i].split('|')[0])

    src = VQApath[:-4]

    if not os.path.exists(dest + 'train'):
        os.mkdir(dest + 'train')
        for img in trainImgList:
            shutil.copy(src + '/organizedData/' + img + '/' + img + '.jpg', dest + 'train')
    with open(dest + 'train.txt', 'w') as f:
        f.write(trainRes)

    if not os.path.exists(dest + 'valid'):
        os.mkdir(dest + 'valid')
        for img in validImgList:
            shutil.copy(src + '/organizedData/' + img + '/' + img + '.jpg', dest + 'valid')
    with open(dest + 'valid.txt', 'w') as f:
        f.write(validRes)

    if not os.path.exists(dest + 'test'):
        os.mkdir(dest  + 'test')
        for img in testImgList:
            shutil.copy(src + '/organizedData/' + img + '/' + img + '.jpg', dest + 'test')
    with open(dest + 'test.txt', 'w') as f:
        f.write(testRes)

    return 0

def moveVQApathToDest(file_path,save_dir,dataset):
    # 因为 file_path 里面没有文件夹，所以不处理有文件夹的情况
    pathDir = os.listdir(file_path)  # os.listdir(file_path) 是获取指定路径下包含的文件或文件夹列表
    for filename in pathDir:  # 遍历pathDir下的所有文件filename
        print(filename)
        from_path = os.path.join(file_path, filename)  # 旧文件的绝对路径(包含文件的后缀名)
        to_path = save_dir + "\\" + dataset  # 新文件的绝对路径

        if not os.path.isdir(to_path):  # 如果 to_path 目录不存在，则创建
            os.makedirs(to_path)
        shutil.copy(from_path, to_path)


if __name__ == '__main__':
    # after labeling 1. extract labelled information from ct_validation 2. generate VQA pairs
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='test')
    parser.add_argument('--dest')
    args = parser.parse_args()
    dataset = args.name  # dataset name without route
    dest = args.dest
    print(dataset)

    VQApath = "/data2/entity/bhy/VQADEMO/demo/src/main/resources/uploadedDataset/" + dataset + "/VQA/"
    # write = mysqlConnector.TestMysql()
    # train, valid, test = mysqlConnector.getPro(dataset)
    # csv_path = write.write(VQApath[:-1], dataset)
    #
    # if not os.path.exists(dest+'/'+dataset):
    #     os.mkdir(dest+'/'+dataset)
    # generateVQA(csv_path,dest+'/'+dataset+'/',train, valid, test)
    mysqlConnector.setDatasetStatus(dataset, VQApath)
    # moveVQApathToDest(VQApath,dest,dataset)


