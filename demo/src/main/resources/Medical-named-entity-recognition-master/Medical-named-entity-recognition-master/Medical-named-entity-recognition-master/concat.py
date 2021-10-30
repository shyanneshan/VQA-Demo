
#coding:utf-8
import json

# testdata=[ ('：', 'BODY-B'), ('缘', 'BODY-I'), ('于', 'BODY-I'), ('2', 'BODY-I'), ('0', 'BODY-I'), ('1', 'BODY-I'), ('3', 'SIGNS-B'), ('-', 'SIGNS-I'), ('0', 'O'), ('6', 'O'), ('-', 'O'), ('2', 'O'), ('7', 'BODY-B'), ('，', 'BODY-I'), ('1', 'BODY-I'), ('5', 'O'), (':', 'O'), ('0', 'O'), ('0', 'BODY-B'), ('左', 'BODY-I'), ('右', 'CHECK-B'), ('从', 'CHECK-I'), ('高', 'O'), ('约', 'CHECK-B'), ('6', 'CHECK-I'), ('m', 'CHECK-I'), ('处', 'CHECK-I'), ('不', 'CHECK-I'), ('慎', 'CHECK-I'), ('坠', 'O'), ('落', 'O'), ('，', 'O'), ('当', 'O'), ('即', 'O'), ('人', 'O'), ('事', 'O'), ('不', 'O'), ('省', 'O'), ('，', 'CHECK-B'), ('呼', 'CHECK-I'), ('之', 'O'), ('不', 'O'), ('应', 'O'), ('，', 'O'), ('伤', 'O'), ('后', 'BODY-B'), ('伴', 'BODY-I'), ('口', 'BODY-I'), ('鼻', 'BODY-I'), ('腔', 'O'), ('流', 'O'), ('血', 'O'), ('。', 'O'), ('感', 'O'), ('头', 'O'), ('痛', 'O'), ('、', 'BODY-B'), ('头', 'BODY-I'), ('晕', 'BODY-I'), ('，', 'O'), ('不', 'O'), ('能', 'O'), ('回', 'O'), ('忆', 'O'), ('受', 'O'), ('伤', 'BODY-B'), ('经', 'BODY-I'), ('过', 'O'), ('。', 'O'), ('由', 'O'), ('家', 'CHECK-B'), ('属', 'CHECK-I'), ('急', 'CHECK-I'), ('送', 'BODY-I'), ('我', 'O'), ('院', 'O'), ('，', 'O'), ('急', 'BODY-B'), ('诊', 'BODY-I'), ('行', 'O'), ('头', 'O'), ('胸', 'O'), ('腹', 'O'), ('部', 'O'), ('C', 'O'), ('T', 'O'), ('检', 'O'), ('查', 'DISEASE-B'), ('示', 'DISEASE-I'), ('：', 'DISEASE-I'), ('颅', 'DISEASE-I'), ('内', 'DISEASE-I'), ('未', 'O'), ('见', 'O'), ('明', 'O'), ('显', 'O'), ('血', 'O'), ('肿', 'O'), ('，', 'O'), ('气', 'O'), ('颅', 'O'), ('，', 'O'), ('左', 'O'), ('眶', 'O'), ('后', 'O'), ('少', 'O'), ('量', 'O'), ('积', 'O'), ('气', 'O'), ('，', 'O'), ('颅', 'O'), ('底', 'O'), ('骨', 'O'), ('折', 'O'), ('累', 'BODY-B'), ('及', 'BODY-I'), ('右', 'O'), ('眶', 'O'), ('上', 'O'), ('壁', 'BODY-B'), ('、', 'BODY-I'), ('蝶', 'O'), ('骨', 'O'), ('体', 'O')]
# beginList=['TREATMENT-B','BODY-B','SIGNS-B','CHECK-B','DISEASE-B']
#
#
#
# def concatRes(testdata):
#     new_list = []
#     for i in range(len(testdata)):
#         data = testdata[i]
#         entity_type = data[1][:-2]
#         if data[1] in beginList:
#             left = i
#             right = i
#             currentRes = data[0]
#             for j in range(i + 1, len(testdata)):
#                 currentType = testdata[j][1]
#                 if currentType != entity_type + '-I':
#
#                     right = j + 1
#                     break
#                 else:
#                     currentRes=currentRes+testdata[j][0]
#             if left != right:
#                 tmp = [currentRes, entity_type]
#                 new_list.append(tmp)
#
#             i = j
#     print(new_list)
#     return new_list
#
# concatRes(testdata)
#
# dict = {'CHECK': [],
#         'SIGNS': [],
#         'DISEASE': [],
#         'TREATMENT': [],
#         'BODY': []
#         }
# result=[['颅', 'BODY'], ['血肿', 'SIGNS'], ['气颅', 'SIGNS'], ['左眶', 'BODY'], ['折累', 'SIGNS'], ['右眶上壁', 'BODY'], ['蝶骨体', 'BODY'], ['鼻骨', 'BODY'], ['鼻中隔', 'BODY'], ['左侧上颌窦', 'BODY'], ['蝶窦', 'BODY'], ['双侧筛窦', 'BODY'], ['左颞部软组织', 'BODY'], ['肿胀', 'SIGNS'], ['右眼', 'BODY'], ['两肺', 'BODY'], ['挫伤', 'CHECK'], ['上腹部SCT', 'CHECK'], ['X线', 'CHECK'], ['右股骨中段', 'BODY'], ['骨折', 'SIGNS']]
# for r in result:
#
#     dict[r[1]].append(r[0])
# print(dict)

# with open("data_new7.16/test.txt") as f:
#     total = []
#     res=''
#     for line in f:
#         # print(line)
#         if len(line.strip()) == 1 or line[0] == " " or line=='\n':
#             continue
#         s = line.strip('\n').split(' ')
#         s[1].strip("\t")
#         # print("s1:",s[1])
#         try:
#             if s[1] == 'B-DISEASE':
#                 s[1] = 'DISEASE-B'
#             elif s[1] == 'B-BODY':
#                 s[1] = 'BODY-B'
#             elif s[1] == 'B-SYMPTOM':
#                 s[1] = 'SIGNS-B'
#             elif s[1] == 'B-CHECK':
#                 s[1] = 'CHECK-B'
#             elif s[1] == 'B-TREATMENT':
#                 s[1] = 'TREATMENT-B'
#             elif s[1] == 'I-DISEASE':
#                 s[1] = 'DISEASE-I'
#             elif s[1] == 'I-BODY':
#                 s[1] = 'BODY-I'
#             elif s[1] == 'I-SYMPTOM':
#                 s[1] = 'SIGNS-I'
#             elif s[1] == 'I-CHECK':
#                 s[1] = 'CHECK-I'
#             elif s[1] == 'I-TREATMENT':
#                 s[1] = 'TREATMENT-I'
#         except:
#             print(Exception)
#         total.append(' '.join(s) + '\n')
#         res=res+'\t'.join(s) + '\n'
#     # print(total)' '.join(s) + '\n'
#     str=''
#     str.join(total)
#     with open("test.txt",'w') as f:
#         # print(res)
#         f.write(res)

def extract_originaltext():
    res=''
    with open("test-origin.txt",'r') as f:
        lines=f.readlines()
        for line in lines:
            s=line.split(" ")


            if s[0]=='。'or s[0]=='；'or s[0]=='：'or s[0]=='，' or s[0]=='|':
                res=res+s[0]+'\n'
            else:
                res=res+s[0]


        print(res)
        with open('requirement.txt','w') as f2:
            f2.write(res)


def new():
    with open('requirement.txt', 'r', encoding='utf-8') as f:
        with open('train.txt', 'w') as f2:
            lines = f.readlines()
            for line in lines:
                if line.split():
                    f2.writelines(line)
# extract_originaltext()
#
# new()
with open('/home/wxl/Documents/named_entity_recognition-master/ResumeNER/test.char.bmes', 'r', encoding='utf-8') as f:
    with open('test1.char.bmes', 'w') as f2:
        lines = f.readlines()
        for line in lines:
            if line.split():
                f2.writelines(line)

with open('test1.char.bmes', 'r') as f2:
    lines = f2.readlines()
    res=""
    for line in lines:
        s=line.split(" ")

        if s[1]=="I-DISEASE\n" or s[1]=="B-DISEASE\n":
            res+=line
        else:
            res=res+s[0]+" O\n"

    with open('test2.char.bmes','w+') as f3:
        f3.writelines(res)