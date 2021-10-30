path=r'D:\program\vqamed2019\VQA-Med-2019\VQAMed2019Test\VQAMed2019_Test_Questions_w_Ref_Answers.txt'
with open('test.txt','w',encoding='utf-8') as fw:
    res=""
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            res+=line.split('|')[0]+'|'+line.split('|')[2]+'|'+line.split('|')[3]
    fw.write(res)
