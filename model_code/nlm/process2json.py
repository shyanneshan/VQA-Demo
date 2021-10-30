import json

# paths=['dataset/train.txt','dataset/val.txt','dataset/test.txt']
#
#
# for path in paths:
#     id=1
#     target_path='dataset/'+path.split('/')[-1].split('.')[0]+'.json'
#     res=[]
#     with open(path,'r',encoding='utf-8') as f:
#         for line in f:
#             s=line.split('|')
#             img_id=s[0]
#             question=s[1]
#             answer=s[2]
#             d={"question":question,"question_id":id,"answer":answer,"image_id":img_id}
#             id+=1
#             res.append(d)
#     with open(target_path,'w',encoding='utf-8') as fw:
#         json.dump(res, fw)

def process2json(args):
    datapath=args.data_dir
    paths = [datapath+'/train.txt', datapath+'/val.txt', datapath+'/test.txt']

    for path in paths:
        id = 1
        target_path = datapath+'/' + path.split('/')[-1].split('.')[0] + '.json'
        res = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.split('|')
                img_id = s[0]
                question = s[1]
                answer = s[2]
                d = {"question": question, "question_id": id, "answer": answer, "image_id": img_id}
                id += 1
                res.append(d)
        with open(target_path, 'w', encoding='utf-8') as fw:
            json.dump(res, fw)
