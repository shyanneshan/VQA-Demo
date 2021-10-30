#!/usr/bin/Python
# -*- coding: utf-8 -*
import argparse
import os

import nltk
from rake_nltk import Rake
import json

def readWiki():
    # wikilist=[]
    with open('wiki/wiki_order.json','r',encoding='utf-8') as f:
        obj=json.load(f)
    return obj



# sort method
# def takeKey(elem):
#     return elem["title"]

# wiki_json.sort(key=takeKey)

# print(wiki_json)
# with open('wiki/wiki_order.json','w',encoding='utf-8') as f:
#     json.dump(wiki_json,f,indent=4)

def getRelatedWord(sent):
    r = Rake()
    r.extract_keywords_from_text(sent)
    res=[]
    for w in r.get_ranked_phrases_with_scores():
        res.append(w[1])
    tokens = nltk.word_tokenize(sent.lower())
    text = nltk.Text(tokens)
    tags = nltk.pos_tag(text)
    rres=[]
    for w in tags:
        if (w[1]=='NN' or w[1]=='JJ' ) and (w in res):
            rres.append(w)
    if len(rres)==0 and len(res)==0:
        return res
    if len(res)==0:
        return [rres[0]]
    if len(rres)==0:
        return [res[0]]




def index_withoutexception(word,words):
    try:
        return words.index(word)
    except:
        return -1

def searchInWiki(word,wiki_words):
    idx=index_withoutexception(word,wiki_words)
    # return idx
    if idx!=-1:
        return idx
    for i in range(len(wiki_words)):
        wiki_words_list=wiki_words[i].split()
        if word == wiki_words_list[0]: # the first word in wiki title is referred word is needed
            return i
    return -1

def createQuery(dataset_path,mode,wiki_json,wiki_words):
    target_dic = os.path.join(dataset_path, "query_" + mode)
    if not os.path.exists(os.path.join(dataset_path,"query_"+mode)):
        os.mkdir(target_dic)

    with open(os.path.join(dataset_path,mode+'_an.json'),'r',encoding='utf-8') as f:
        obj=json.load(f)

    id=0
    for item in obj:
        id=item["question_id"]
        question=item["question"]
        # print(id,question)
        i=1
        words=getRelatedWord(item["question"])
        for w in words:
            name_path = str(id) + '.00'+str(i)+'.json'
            idx=searchInWiki(w,wiki_words)
            if idx!=-1:
                target_path=os.path.join(target_dic,name_path)
                wiki_pair=wiki_json[idx]
                title=wiki_pair['title']
                doc=wiki_pair['abstract']
                if doc=='':
                    # i=1
                    continue
                # write to target file
                with open(target_path, 'w', encoding='utf-8') as fw:
                    json.dump({"title":title,
                               "doc":doc,
                               "question":question},fw,indent=4)
                i+=1

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default="dataset", help='dataset origin folder.')

    args = parser.parse_args()
    root=args.dataset
    modes=['train','val','test']
    #make wiki words
    wiki_words = []
    wiki_json = readWiki()
    for item in wiki_json:
        wiki_words.append(item['title'].lower())
    for mode in modes:
        createQuery(root,mode,wiki_json,wiki_words)

