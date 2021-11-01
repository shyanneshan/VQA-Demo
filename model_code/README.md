# models Code

**This is the code for five models.**

#### Joint embedding model

##### ODL

This model is open source [here]（https://github.com/aioz-ai/MICCAI19-MedVQA）.

maml and autoEncoder weight are placed in /ODL/pretrained

glove file is placed in /ODL/glove/glove.6B.300d.txt

#### Encoder-Decoder model

##### NLM

This model is open source [here]（https://github.com/sarrouti/VQA）.

word2vector embedding file is placed in /nlm/.vector_cache/PubMed-w2v.txt. Can be found online.

if you want to use glove embedding, use **--embedding-name '6B'**

##### Vgg-Seq2Seq

This model is open source [here]（https://github.com/bashartalafha/VQA-Med）.

glove file is placed in /vgg-seq2seq/glove/glove.6B.300d.txt

#### Attention-based model

##### MMBERT

This model is open source [here]（https://github.com/VirajBagal/MMBERT）.

#### Knowledge embedding model

##### ArticleNet

This model is open source [here]（https://github.com/Adam1679/mutan-article-net）

wiki knowledge base file is placed in /ArticleNet/wiki/wiki_order.json

We download wiki knowledge xml file. Get the wiki_order.json file about its title and content， and sort. he processing file have been uploaded.

glove file is placed in /ArticleNet/glove/glove.6B.50d.txt


