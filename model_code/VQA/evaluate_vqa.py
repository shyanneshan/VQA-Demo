from torchvision import transforms
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
        if torch.cuda.is_available():
            images = images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
        qlengths = process_lengths(questions)
        qlengths.sort(reverse = True)
        # Predict.
        outputs = vqa.predict_from_question(images, questions, qlengths)
        
        for i in range(images.size(0)):
            #print (images[i])
            output = vocab.tokens_to_words(outputs[i])
            preds.append(output)

            question = vocab.tokens_to_words(answers[i])
            gts.append(question)
        bar.update(iterations)
    scores = nlge.compute_metrics(ref_list=[gts], hyp_list=preds)
    return scores, gts, preds
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
    vocab = load_vocab(params.project_path+'/'+params.vocab_path)

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
    data_loader = get_loader(args.val_dataset, transform,
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
    vqa.load_state_dict(torch.load(args.model_path+"vqa-tf-"+str(args.num_epochs)+".pkl"))

    # Setup GPUs.
    if torch.cuda.is_available():
        logging.info("Using available GPU...")
        vqa.cuda()

    scores, gts, preds = evaluate(vqa, data_loader, vocab, args, params)

    # Print and save the scores.
    print (scores['Bleu_2'])
