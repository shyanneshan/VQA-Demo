
# 去隐私？
# 预处理，去掉无
# 用模型跑
import os
import jieba


jieba.add_word("个人史", freq=30)
jieba.add_word("婚育史", freq=40)
jieba.add_word("婚姻史", freq=40)
jieba.add_word("家族史", freq=50)
jieba.add_word("姓名", freq=50)


src = r'C:\Users\shyanne\Desktop\dataset\datasetGene\data\src'

res = r'C:\Users\shyanne\Desktop\dataset\datasetGene\data\res'


# 处理无后面的症状，直接舍弃
def doNotHave(list):
    for i in range(len(list)):
        if list[i] == "无" or list[i] == "否认" :
            list[i] = -1
            for j in range(i, len(list)):
                if list[j] != '，':
                    list[j] = -1
                else:
                    list[j] = -1;
                    break
    while (-1 in list):
        list.remove(-1)
    return list

def removePrivacy(src , dest):
    file1 = open(src, 'r', encoding='utf-8') # 要去掉隐私的文件
    file2 = open(dest, 'w', encoding='utf-8') # 生成没有隐私信息的文件
    try:
        for line in file1.readlines():
            if line == '\n':
                line = line.strip("\n")
            line=line.lstrip()
            line = line.replace('/t', '')
            line = line.replace(' ', '')

            seg_list = list(jieba.cut(line, cut_all=False))
            if "个人史" in seg_list or "婚育史" in seg_list or "婚姻史" in seg_list or "家族史" in seg_list or "个人史" in seg_list :
                continue
            if "姓名" in seg_list:
                break
            print(seg_list)

            seg_result=doNotHave(seg_list)

            file2.write(''.join(seg_result))


    finally:
        file1.close()
        file2.close()





if __name__ == '__main__':
    for dic in os.listdir(src):
        dicname = src + '\\' + dic
        destname=res+'\\'+dic
        removePrivacy(dicname,destname)


