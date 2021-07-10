import json
import os
import zipfile

from PreProcess import removePrivacy
from lstm_predict import predictAll

src='/home/wxl/Documents/VQADEMO/demo/src/main/resources/datasetOriginalData/7.9.zip'
dest='/home/wxl/Documents/VQADEMO/demo/src/main/resources/datasetOriginalData/res'
zipDest='/home/wxl/Documents/VQADEMO/demo/src/main/resources/datasetOriginalData/'

def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


def generateQuestions(dicname,destname):
    imgname = dicname + '/img'
    textname = dicname + '/text.txt'
    othername = dicname + '/other.txt'
    textPreProcessedName = destname + '/preprocessed.txt'
    removePrivacy(textname, textPreProcessedName)

    NERinputName = textPreProcessedName
    NERoutputName = destname + '/entity.json'
    predictAll(NERinputName, NERoutputName)
    resultEntity = destname + '/ResultEntity.json'
    QA = destname + '/QA.txt'
    # 将类文件对象中的JSON字符串直接转换成 Python 字典
    with open(NERoutputName, 'r', encoding='utf-8') as f:
        ret_dic = json.load(f)
        with open(othername, 'r', encoding='utf-8') as f2:
            data = f2.readlines()
            for d in data:
                da = d.split("|")
                if da[1] == "药疗":
                    ret_dic['TREATMENT'].append(da[2])
                elif da[1] == "检查" or da[1] == "检验":
                    ret_dic['CHECK'].append(da[2])

            with open(resultEntity, "w", encoding='utf-8') as f:
                # indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
                f.write(json.dumps(ret_dic, indent=4))
    os.remove(NERoutputName)
    os.remove(textPreProcessedName)

    with open(resultEntity, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        for data in json_data.values():
            print(data)
            i = 0
            while (i != len(data)):
                if len(data[i]) == 1:
                    data.pop(i)
                    i = i - 1
                i = i + 1
            data = list(set(data))
            print(data)
        QA = generateDISnCHECK(json_data["DISEASE"], json_data["CHECK"])
        print(QA)
        with open(QA, 'w', encoding='utf8') as f:
            for i in QA:
                f.write(i + '\n')
    os.remove(resultEntity)


# input : resultEntity
# output : questions
def generateDISnCHECK(dis,check):
    dis=','.join(dis)
    check=','.join(check)

    question='我有'+dis+',可以进行什么检查？'
    answer='可以考虑'+check
    return [question,answer]



def main(src,dest):
    zipDest=src[:-4]
    print(zipDest)
    unzip_file(src, zipDest)

    for dic in os.listdir(zipDest):
        dicname = zipDest + '/' + dic
        destname=dest+'/'+dic
        if not dic in os.listdir(dest):
            os.makedirs(destname)
        print(dicname)
        print(destname)
        generateQuestions(dicname,destname)


main(src,dest)


