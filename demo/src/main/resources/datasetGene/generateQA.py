import os
import shutil

from ner import ner

def generateDISnCHM(dic):
    QAlist = []
    for x in dic['DISEASE']:
        for y in dic['CHEMICAL']:
            q = 'I have ' + x + ', what treatment is available?'
            a = y
            QAlist.append([q,a])
    return QAlist


def generateQA(src, dest, train, valid, test):
    result = []
    trainRes = ''
    validRes = ''
    testRes = ''
    for dic in os.listdir(src):
        dicname = src + '/' + dic
        for i in os.listdir(dicname):
            # os.path.splitext():分离文件名与扩展名
            if os.path.splitext(i)[1] == '.jpg':
                imgname = dicname + '/' + i
                imgid = i.split('.')[0]
            elif os.path.splitext(i)[1] == '.txt':
                textname = dicname + '/' + i

        print(os.listdir(dicname))
        with open(textname, 'r') as f:
            data = f.read()
            entities = ner(data)
            entDic={'DISEASE':[],'CHEMICAL':[]}
            for entity in entities:
                if entity[1]=='DISEASE':
                    entDic['DISEASE'].append(entity[0])
                elif entity[1]=='CHEMICAL':
                    entDic['CHEMICAL'].append(entity[0])
            QAlist = generateDISnCHM(entDic)
            for QA in QAlist:
                result.append(imgid + '|' + QA[0] + '|' + QA[1])

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

    if not os.path.exists(dest ):
        os.mkdir(dest)

    if not os.path.exists(dest + 'train'):
        os.mkdir(dest + 'train')
        for img in trainImgList:
            shutil.copy(src + '/' + img + '/' + img + '.jpg', dest + 'train')
    with open(dest + 'train.txt', 'w') as f:
        f.write(trainRes)

    if not os.path.exists(dest + 'valid'):
        os.mkdir(dest + 'valid')
        for img in validImgList:
            shutil.copy(src + '/' + img + '/' + img + '.jpg', dest + 'valid')
    with open(dest + 'valid.txt', 'w') as f:
        f.write(validRes)

    if not os.path.exists(dest + 'test'):
        os.mkdir(dest + 'test')
        for img in testImgList:
            shutil.copy(src + '/' + img + '/' + img + '.jpg', dest + 'test')
    with open(dest + 'test.txt', 'w') as f:
        f.write(testRes)

    return 0

# generateQA('/data2/entity/bhy/VQADEMO/demo/src/main/resources/uploadedDataset/2.txt')