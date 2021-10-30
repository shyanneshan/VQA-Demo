'''
This document is used to process wiki database xml file.
The structure of the xml is as follow:
<feed>
<doc>
<title>Wikipedia: Anarchism</title>
<url>https://en.wikipedia.org/wiki/Anarchism</url>
<abstract>Anarchism is a political philosophy and movement that is sceptical of authority and rejects all involuntary, coercive forms of hierarchy. Anarchism calls for the abolition of the state, which it holds to be undesirable, unnecessary, and harmful.</abstract>
<links>
<sublink linktype="nav"><anchor>Etymology, terminology, and definition</anchor><link>https://en.wikipedia.org/wiki/Anarchism#Etymology,_terminology,_and_definition</link></sublink>
</links>
</doc>
</feed>
'''

# !/usr/bin/python
# -*- coding: UTF-8 -*-
# coding=utf-8
import ixml
from io import StringIO
import json
import os
res_set=set()
if os.path.exists('wiki/wiki.json'):
    os.remove('wiki/wiki.json')


res = []
# data=open('wiki/test.xml','rb')
data = open("wiki/enwiki-latest-abstract.xml", 'rb')
languages, countries = set(), set()
res_dic = {"title": "", "abstract": ""}

for path, event, value in ixml.parse(data):
    flag = False  # if abstract is got, flag=True; or flag=False
    if path == 'feed.doc.title':
        # languages.add(value)
        res_dic["title"] = value.split(':')[1][1:]
        # print("title ", value)
    elif path == 'feed.doc.abstract':
        # countries.add(value)
        res_dic["abstract"] = value
        flag = True
        # print("abstract ", value)
    if flag:
        res.append(res_dic)
        if str(res_dic) in res_set:
            print("has this ",str(res_dic))
            continue
        # res_set.add(str(res_dic))
        print(res_dic)
        res_dic = {"title": "", "abstract": ""}
with open('wiki/wiki.json','w+',encoding='utf-8') as fw:
    json.dump(res, fw, indent=4)
# print(languages)
# print(countries)
