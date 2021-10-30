import argparse
import json
import os


def readtxt(path):
    questions=[]
    answers=[]
    images=[]
    question_ids=[]
    answers_occurence=[]
    mode=path.split('.')[0]
    with open(path,'r',encoding='utf-8') as f:
        question_id=1
        for line in f:
            s=line.split('|')
            imag_name=s[0]
            question=s[1]
            answer=s[2]
            if answer[-1]=='\n':
                answer=answer[:-1]
            question_ids.append(question_id)
            questions.append(question)
            answers.append(answer)
            images.append(imag_name+'.jpg')
            answers_occurence.append([[answer,10]])
            question_id+=1
    return question_ids,questions,answers,images,answers_occurence


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="dataset", help='dataset origin folder.')

    args = parser.parse_args()
    root=args.dataset
    modes=['train','val','test']
    for mode in modes:
        res=[]
        question_ids,questions,answers,images,answers_occurence=readtxt(os.path.join(root,mode+'.txt'))
        for i in range(len(question_ids)):
            question_id=question_ids[i]
            question=questions[i]
            answer=answers[i]
            imag=images[i]
            answer_occurence=answers_occurence[i]
            res_dic={"question_id": question_id, "image_name": imag,
                 "question": question,
                 "answers_occurence": answer_occurence, "answer": answer}

            res.append(res_dic)
        with open(os.path.join(root,mode+'.json'),'w',encoding='utf-8') as fw:
            json.dump(res,fw,indent=4)