#coding:utf-8
testdata=[('现', 'SIGNS-I'), ('病', 'O'), ('史', 'O'), ('：', 'O'), ('缘', 'BODY-I'), ('于', 'BODY-I'), ('入', 'BODY-I'), ('院', 'BODY-I'), ('4', 'O'), ('小', 'O'), ('时', 'O'), ('前', 'SIGNS-B'), ('由', 'SIGNS-I'), ('3', 'O'), ('层', 'O'), ('楼', 'O'), ('高', 'O'), ('处', 'O'), ('坠', 'O'), ('落', 'BODY-I'), ('，', 'BODY-I'), ('当', 'BODY-I'), ('时', 'O'), ('即', 'O'), ('致', 'O'), ('腰', 'DISEASE-B'), ('背', 'DISEASE-I'), ('部', 'DISEASE-I'), ('、', 'DISEASE-I'), ('右', 'DISEASE-I'), ('大', 'DISEASE-I'), ('腿', 'DISEASE-I'), ('肿', 'DISEASE-I'), ('痛', 'DISEASE-I'), ('，', 'DISEASE-I'), ('伴', 'DISEASE-I'), ('右', 'DISEASE-I'), ('髋', 'DISEASE-I'), ('部', 'DISEASE-I'), ('疼', 'DISEASE-I'), ('痛', 'O'), ('、', 'O'), ('活', 'O'), ('动', 'O'), ('受', 'O'), ('限', 'O'), ('，', 'O'), ('伤', 'O'), ('时', 'O'), ('伤', 'O'), ('后', 'O'), ('未', 'O'), ('行', 'O'), ('特', 'O'), ('殊', 'O'), ('处', 'O'), ('理', 'O'), ('，', 'O'), ('就', 'O'), ('诊', 'O'), ('我', 'O'), ('院', 'O'), ('，', 'BODY-I'), ('行', 'O'), ('X', 'O'), ('线', 'O'), ('片', 'O'), ('示', 'O'), ('：', 'O'), ('1', 'O'), ('、', 'O'), ('腰', 'O'), ('1', 'O'), ('椎', 'O'), ('体', 'O'), ('压', 'O'), ('缩', 'O'), ('骨', 'O'), ('折', 'O')]
beginList=['TREATMENT-B','BODY-B','SIGNS-B','CHECK-B','DISEASE-B']



def concatRes(testdata):
    new_list = []
    for i in range(len(testdata)):
        data = testdata[i]
        entity_type = data[1][:-2]
        if data[1] in beginList:
            left = i
            right = i
            currentRes = data[0]
            for j in range(i + 1, len(testdata)):
                currentType = testdata[j][1]
                if currentType == entity_type + '-I':
                    currentRes = currentRes + testdata[j][0]
                    right = j + 1
            if left != right:
                tmp = [currentRes, entity_type]
                new_list.append(tmp)

            i = j
    print(new_list)
    return new_list

concatRes(testdata)