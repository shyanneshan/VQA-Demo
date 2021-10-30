import argparse
import json
import os
from string import punctuation

import sklearn
import torch

from createQuery import createQuery, readWiki
from dataLoader import get_dataloader, reduce_data
from model import ArticleNet
# config = configparser.ConfigParser()
# config.read('config.ini')
from preprocess import loadData, processImag, saveLabel, saveData
from process_documents import process_one_document
from txt2json import readtxt

model_root = 'data/save'


def save_checkpoint(model_dict, model_path):
    torch.save(model_dict, model_path)
    return

def cointinuous2binary(l):
    for i in range(len(l)):
        max_value = max(l[i])
        for j in range(len(l[i])):
            if l[i][j] == max_value:
                l[i][j] = 1
            else:
                l[i][j] = 0
    return l

def train(args):

    model = ArticleNet(args).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bceloss = torch.nn.BCELoss()
    train_loader = get_dataloader(args.dataset,'train', shuffle=True, batch_size=args.batch_size)
    # val_loader = get_dataloader('val', shuffle=False, batch_size=config['val_size'])
    val_loader = get_dataloader(args.dataset,'val', shuffle=False, batch_size=args.batch_size)
    test_loader = get_dataloader(args.dataset,'test', shuffle=False, batch_size=args.batch_size)


    for epoch in range(args.num_epoch):
        #=========================== train =====================
        model.train()
        # with open('artpredict.txt', 'a+')as file_handle:
        #             file_handle.write('epoch:'+str(epoch) + '\n')
        # print("epoch: ",str(epoch))
        for qbatch, vbatch, tbatch, sbatch, ybatch, len_batch in train_loader:
            ypred = model.forward(qbatch, vbatch, tbatch, sbatch, len_batch,args)
            optimizer.zero_grad()

            ypred=torch.hstack((ypred[0],ypred[1],ypred[2]))

            loss = bceloss(ypred, ybatch)

            loss.backward()
            optimizer.step()
        #=========================== eval =====================
        # print("eval..............")
        model.eval()

        total_pred=[]
        total_gt=[]
        with torch.no_grad():
            for qbatch, vbatch, tbatch, sbatch, ybatch, len_batch in val_loader:
                
                a_title, a_sent, a_art = model.forward(qbatch, vbatch, tbatch, sbatch, len_batch,args)

                #分别计算a_title, a_sent, a_art
                # y_title=ybatch[:,0]
                y_sent=ybatch[:,1:-1]
                # y_art=ybatch[:,-1]

                # a_sent = np.rint(a_sent)
                a_sent=cointinuous2binary(a_sent)
                # print(a_sent)

        save_checkpoint(model.state_dict(), args.model_dir)

    sent_right_num = 0
    sent_num = 0
    for qbatch, vbatch, tbatch, sbatch, ybatch, len_batch in test_loader:

        a_title, a_sent, a_art = model.forward(qbatch, vbatch, tbatch, sbatch, len_batch, args)

        # 分别计算a_title, a_sent, a_art
        y_title = ybatch[:, 0]
        y_sent = ybatch[:, 1:-1]
        y_art = ybatch[:, -1]

        # a_sent = np.rint(a_sent)
        a_sent = cointinuous2binary(a_sent)
        a_title = cointinuous2binary(a_title)
        a_art = cointinuous2binary(a_art)

        total_gt.append(y_sent)
        total_pred.append(a_sent)
    PREDS = torch.cat(total_pred).cpu().detach().numpy()
    TARGETS = torch.cat(total_gt).cpu().detach().numpy()
    report = sklearn.metrics.classification_report(TARGETS, PREDS, labels=None, target_names=None,
                                                    sample_weight=None, digits=4, output_dict=False,
                                                    zero_division='warn')
    weighted = report.split('\n')[-2].split('     ')[1].split('    ')  # [1:4]

    res = []
    for w in weighted:
        res.append(str(round(float(w) * 10000)))
    print(" ".join(res))
    # return " ".join(res)
    # print("sent_result:", sent_result)



if __name__=='__main__':
    # print("start")
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="dataset", help='train on dataset')
    parser.add_argument('--model_dir', default="save/dataset.pt", help='model save path.')
    parser.add_argument('--num-epoch', type=int, default=20, help='epoch number')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--q_len', default=50)
    parser.add_argument('--q_hidden', default=128)
    parser.add_argument('--q_layer', default=1)
    parser.add_argument('--tit_len', default=50)
    parser.add_argument('--tit_hidden', default=128)
    parser.add_argument('--t_layer', default=1)
    parser.add_argument('--sent_len', default=50)
    parser.add_argument('--sent_hidden', default=128)
    parser.add_argument('--s_layer', default=1)
    parser.add_argument('--art_hidden', default=128)
    parser.add_argument('--title_num_class', default=1)
    parser.add_argument('--sent_num_class', default=1100)
    parser.add_argument('--art_num_class', default=1)

    args = parser.parse_args()

    root = args.dataset
    modes = ['train', 'val', 'test']
    #txt2json
    # print("txt2json")
    for mode in modes:
        res = []
        question_ids, questions, answers, images, answers_occurence = readtxt(os.path.join(root, mode + '.txt'))
        for i in range(len(question_ids)):
            question_id = question_ids[i]
            question = questions[i]
            answer = answers[i]
            imag = images[i]
            answer_occurence = answers_occurence[i]
            res_dic = {"question_id": question_id, "image_name": imag,
                       "question": question,
                       "answers_occurence": answer_occurence, "answer": answer}

            res.append(res_dic)
        with open(os.path.join(root, mode + '_an.json'), 'w', encoding='utf-8') as fw:
            json.dump(res, fw, indent=4)
    # createQuery
    # print("createQuery")
    #make wiki words
    wiki_words = []
    wiki_json = readWiki()
    for item in wiki_json:
        wiki_words.append(item['title'].lower())
    for mode in modes:
        createQuery(root,mode,wiki_json,wiki_words)
    # process document
    for mode in modes:
        import re, tqdm

        pat = re.compile("(\d+).(\d+).json")
        query_root = os.path.join(root, 'query_' + mode)
        # val_root = "data/query_val"
        given_data = os.path.join(root, "input_" + mode + ".txt")
        # val_given_data = "data/okvqa_an_input_val.txt"
        data_path = os.path.join(root, mode + '_an.json')


        def process(data_path, root, given_data):
            with open(data_path, 'r') as f:
                data = json.loads(f.read())

            n_docs = 0
            n_pos_docs = 0
            n_pos_sentences = 0
            n_sentences = 0
            question2answer = {}
            question2all = {}

            for pair in data:
                question2answer[pair['question_id']] = [item[0] for item in pair["answers_occurence"]]
                question2all[pair['question_id']] = pair

            with open(given_data, "w") as ff:
                for file in tqdm.tqdm(os.listdir(root)):
                    if file.endswith(".json"):
                        # import pdb; pdb.set_trace()
                        # print(file)
                        question_id, document_order = re.findall(pat, file)[0]
                        question_id = int(question_id)
                        with open(os.path.join(root, file)) as f:
                            content = json.loads(f.read())

                        title = content['title']
                        doc = content['doc']
                        question = content['question']
                        answers = question2answer[question_id]
                        titleHasAns, sentences_pairs, docHasAns = process_one_document(title, doc, answers)
                        on_sample = {}
                        on_sample['question'] = question
                        on_sample['title'] = title
                        on_sample['titleHasAns'] = titleHasAns
                        on_sample['sentences_pairs'] = sentences_pairs
                        on_sample['docHasAns'] = docHasAns
                        ff.write("{}\n".format(json.dumps(on_sample)))
                        n_docs += 1
                        n_sentences += len(sentences_pairs)
                        n_pos_sentences += sum([i[1] for i in sentences_pairs])
                        n_pos_docs += docHasAns

            # print("stats")
            # print(f"{n_pos_docs}/{n_docs}")
            # print(f"{n_pos_sentences}/{n_sentences}")


        process(data_path, query_root, given_data)
    # preprocess
    # print("preprocess")
    for mode in modes:
        if not os.path.exists(os.path.join(root,mode+"_npy")):
            os.mkdir(os.path.join(os.path.join(root,mode+"_npy")))
        question_emb = []
        title_emb = []
        sentence_emb = []
        imags = []
        labels = []
        translate_table = dict((ord(char), None) for char in punctuation)
        loadData(os.path.join(root, "input_" + mode + '.txt'), question_emb,
                 title_emb, sentence_emb, labels,translate_table)
        saveData(os.path.join(root, mode + "_npy"), question_emb, sentence_emb)
        saveLabel(os.path.join(root, mode + "_npy"), labels, title_emb)
        processImag(mode, root,imags)
    # dataLoader
    # print("dataLoader")

    for mode in modes:
        # reduce_data('val')
        reduce_data(root,mode+"_npy")

    model_dict = train(args)