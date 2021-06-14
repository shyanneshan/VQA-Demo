"""This script is used to evaluate the VQA model.
"""
import sys
sys.path.append('utils')
from torchvision import transforms

import argparse
import json
import logging
import os
import progressbar
import torch

from models import VQA
from utils import NLGEval
from utils import Dict2Obj
from utils import Vocabulary
from utils import get_loader
from utils import load_vocab
from utils import process_lengths
from utils import get_glove_embedding

import gensim
import numpy as np
from torchtext.vocab import Vectors

from utils.store_dataset import create_dataset_predict


def create_original_txt(args,imag_id,question,answer):
    with open(args.name+'/predict.txt','w',encoding='utf-8') as f:
        f.write(imag_id+'|'+question+'|'+answer)
    return   args.name+'/predict.txt'


def get_wv_embedding(name, embed_size, vocab):
    """Construct embedding tensor.

    Args:
        name (str): Which GloVe embedding to use.
        embed_size (int): Dimensionality of embeddings.
        vocab: Vocabulary to generate embeddings.
    Returns:
        embedding (vocab_size, embed_size): Tensor of
            GloVe word embeddings.
    """
    """    for index, w in zip(vocab.values(), vocab.keys()):
        if w in list(word_vecs.wv.vocab):
            vec = model[w]
        else:
            vec = np.random.uniform(-0.25,0.25, embed_size)
        embedding[index] = vec    
    
    glove = torchtext.vocab.GloVe(name=name,
                                  dim=str(embed_size))
    """
    #name='/home/sarroutim2/PosDoc NLM/Question Answering/Embedding and pretained models/wikipedia-pubmed-and-PMC-w2v.txt'
    w2v=Vectors(name=name)##cache='.vector_cache/wiki-PubMed-w2v.txt.pt'
    vocab_size = len(vocab)
    embedding = torch.zeros(vocab_size, embed_size)
    for i in range(vocab_size):
        embedding[i] = w2v[vocab.idx2word[str(i)]]

    return embedding
def evaluate(vqa, data_loader, vocab, args, params):
    """Runs BLEU, METEOR, CIDEr and distinct n-gram scores.

    Args:
        vqa: question generation model.
        data_loader: Iterator for the data.
        args: ArgumentParser object.
        params: ArgumentParser object.

    Returns:
        A float value of average loss.
    """
    vqa.eval()
    nlge = NLGEval(no_glove=True, no_skipthoughts=True)
    preds = []
    gts = []
    bar = progressbar.ProgressBar(maxval=len(data_loader))
    bar.start()
    for iterations, (images, questions, answers,
                      _) in enumerate(data_loader):

        # Set mini-batch dataset
        '''zhu shi hui lai . don't forget it. bai'''
        # if torch.cuda.is_available():
        #     images = images.cuda()
        #     questions = questions.cuda()
        #     answers = answers.cuda()
        qlengths = process_lengths(questions)
        qlengths.sort(reverse = True)


        # Predict.
        # print("images: ",images.shape)
        # print("questions: ",questions.shape)
        # print("qlengths: ",qlengths)
        outputs = vqa.predict_from_question(images, questions, qlengths)

        for i in range(images.size(0)):
            #print (images[i])
            output = vocab.tokens_to_words(outputs[i])
            preds.append(output)

            question = vocab.tokens_to_words(answers[i])
            gts.append(question)
        bar.update(iterations)
    # print ('='*80)
    # print ('GROUND TRUTH')
    # print (gts[:args.num_show])
    # print ('-'*80)
    # print ('PREDICTIONS')
    print (preds[:args.num_show])
    # print ('='*80)
    # scores = nlge.compute_metrics(ref_list=[gts], hyp_list=preds)
    return  preds
    #return gts, preds


def main(args):
    """Loads the model and then calls evaluate().

    Args:
        args: Instance of ArgumentParser.
    """

    # Load the arguments.
    model_dir = os.path.dirname(args.model_path)
    params = Dict2Obj(json.load(
            open(os.path.join(args.model_path,args.run_name, "args.json"), "r")))

    # Config logging
    log_format = '%(levelname)-8s %(message)s'
    logfile = os.path.join(model_dir, 'eval.log')
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(json.dumps(args.__dict__))

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # Load vocabulary wrapper.
    vocab = load_vocab(params.vocab_path)

    # Build data loader
    logging.info("Building data loader...")

    # Load GloVe embedding.
    if params.use_glove:
        embedding = get_glove_embedding(params.embedding_name,
                                        300,
                                        vocab)
    elif params.use_w2v:
        embedding = get_wv_embedding(params.embedding_name,
                                        200,vocab)
    else:
        embedding = None

    # Build data loader
    indices=[]
    ii=0
    for ii in range(500):
        indices.append(ii)
    logging.info("Building data loader...")
    data_loader = get_loader(args.dataset, transform,
                                 args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples,
                                 indices=None)
    logging.info("Done")

    # Build the models
    logging.info('Creating IQ model...')
    vqa = VQA(len(vocab), params.max_length, params.hidden_size,

             vocab(vocab.SYM_SOQ), vocab(vocab.SYM_EOS),
             num_layers=params.num_layers,
             rnn_cell=params.rnn_cell,
             dropout_p=params.dropout_p,
             input_dropout_p=params.input_dropout_p,
             encoder_max_len=params.encoder_max_len,
             embedding=embedding,
             num_att_layers=params.num_att_layers,
             #use_attention=params.use_attention,
             z_size=params.z_size,
             no_question_recon=params.no_question_recon,
             no_image_recon=params.no_image_recon
             )
    logging.info("Done")

    logging.info("Loading model.")
    vqa.load_state_dict(torch.load(args.model_path+"vqa-tf-"+args.state+".pkl"))

    # Setup GPUs.
    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        vqa.cuda()

    evaluate(vqa, data_loader, vocab, args, params)


if __name__ == '__main__':
    from utils import Vocabulary
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='When set, only evalutes that many data points.')
    parser.add_argument('--num-show', type=int, default=50,
                        help='Number of predictions to print.')
    parser.add_argument('--from-answer', type=str, default='true',
                        help='When set, only evalutes iq model with answers;'
                        ' otherwise it tests iq with answer types.')
    parser.add_argument('--state', type=str, default='1',
                        help='Path for saving results.')
    # Data parameters.
    parser.add_argument('--dataset', type=str,
                        default='data/name/pred_dataset_val.hdf5',
                        help='path for train annotation json file')
    #input
    #need to config
    parser.add_argument('--model-path', type=str, default='weights/tf1/'
                        ,help='Path for loading trained models')
    parser.add_argument('--run_name', type=str,
                        default='data/name',
                        help='path for train annotation json file')
    parser.add_argument('--vocab_path', type=str,
                        default='data/name/vocab_vqa.json',
                        help='path for train annotation json file')
    parser.add_argument('--output', type=str,
                        default='data/name',
                        help='path for train annotation json file')
    parser.add_argument('--imagPath', type=str,
                        default='data/name',
                        help='path for predicting imag')
    parser.add_argument('--question', type=str,
                        default='data/name',
                        help='predict question')
    args = parser.parse_args()
    imagName=args.imagPath.split('/')[-1].split('.')[0]
    imagDir="/".join(args.imagPath.split('/')[:-1])
    print(imagDir)
    print(imagName)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    txt_path=create_original_txt(args,imagName,args.question,
                        'cta - ct angiography')
    create_dataset_predict(imagDir, txt_path,args.vocab_path,args.output)
    main(args)
    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()
