import json
import os
from PreProcess import removePrivacy
from lstm_predict import predictAll
src='/home/wxl/Downloads/6.13测试'
dest='/home/wxl/Downloads/6.13result'

def generateQuestions(dicname,destname):
    for dic in os.listdir(dicname):
        imgname = dicname + '/img'
        textname=dicname+'/text.txt'
        othername=dicname+'/other.txt'
        textPreProcessedName=destname+'/preprocessed.txt'
        removePrivacy(textname,textPreProcessedName)

        NERinputName = textPreProcessedName
        NERoutputName = destname  + '/entity.json'
        predictAll(NERinputName,NERoutputName)
        resultEntity=destname  + '/ResultEntity.json'
        # 将类文件对象中的JSON字符串直接转换成 Python 字典
        with open(NERoutputName, 'r', encoding='utf-8') as f:
            ret_dic = json.load(f)
            with open(othername,'r',encoding='utf-8') as f2:
                data = f2.readlines()
                for d in data:
                    da = d.split("|")
                    if da[1]=="药疗":
                        ret_dic['TREATMENT'].append(da[2])
                    elif da[1]=="检查" or da[1]=="检验":
                        ret_dic['CHECK'].append(da[2])
                with open(resultEntity, "w", encoding='utf-8') as f:
                    # indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
                    f.write(json.dumps(ret_dic, indent=4))


def main(src,dest):
    for dic in os.listdir(src):
        dicname = src + '/' + dic
        destname=dest+'/'+dic
        if not dic in os.listdir(dest):
            os.makedirs(destname)
        generateQuestions(dicname,destname)


main(src,dest)


#
#
# # prprocess
# from PreProcess import preprocess
# srcDir=''
# mediateDir1='/home/wxl/Documents/VQADEMO/demo/src/main/resources/testTextData/res'
# mediateDir2='/home/wxl/Documents/VQADEMO/demo/src/main/resources/testTextData/test530'
# #preprocess(srcDir,mediateDir1)
#
# from lstm_predict import predictAll
# predictAll(mediateDir1,mediateDir2)
