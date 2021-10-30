"""This script is used to evaluate the VQA model.
"""
import re

import h5py
import nltk
import numpy as np
from PIL import Image
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

from torchtext.vocab import Vectors
from utils.data_loader import VQADataset, collate_fn
# from utils.store_dataset import save_dataset

def create_questions_images_ids(questions_list):
    """


    Returns:
        questions: set of question ids.
        image_ids: Set of image ids.
    """
    questions = set()
    image_ids = set()
    for q in questions_list:
        # print(q['question_id'])
        question_id = q['question_id']
        questions.add(question_id)
        image_ids.add(q['image_id'])

    return questions, image_ids

def tokenize(sentence):
    """Tokenizes a sentence into words.

    Args:
        sentence: A string of words.

    Returns:
        A list of words.
    """
    if len(sentence) == 0:
        return []
    if isinstance(sentence,str):
        sentence=sentence
    else:
        sentence=sentence.decode('utf-8')
    sentence = re.sub('\.+', r'.', sentence)
    sentence = re.sub('([a-z])([.,!?()])', r'\1 \2 ', sentence)
    sentence = re.sub('\s+', ' ', sentence)

    tokens = nltk.tokenize.word_tokenize(
            sentence.strip().lower())
    return tokens

def process_text(text, vocab, max_length=20):
    """Converts text into a list of tokens surrounded by <start> and <end>.

    Args:
        text: String text.
        vocab: The vocabulary instance.
        max_length: The max allowed length.

    Returns:
        output: An numpy array with tokenized text.
        length: The length of the text.
    """
#    if text==None:
#        text="Initial chest radiograph"
    tokens = tokenize(text.lower().strip())
    output = []
    output.append(vocab(vocab.SYM_SOQ))  # <start>
    output.extend([vocab(token) for token in tokens])
    output.append(vocab(vocab.SYM_EOS))  # <end>
    length = min(max_length, len(output))

    return np.array(output[:length]), length

def save_dataset(image_path, question, choice, vocab, output,
                 im_size=224, max_q_length=20, max_a_length=6, max_c_length=20,
                 with_answers=False):
    """Saves the Visual Genome images and the questions in a hdf5 file.

    Args:
        image_dir: Directory with all the images.
        questions: Location of the questions.
        annotations: Location of all the annotations.
        vocab: Location of the vocab file.
        output: Location of the hdf5 file to save to.
        im_size: Size of image.
        max_q_length: Maximum length of the questions.
        max_a_length: Maximum length of the answers.
        with_answers: Whether to also save the answers.
    """
    # Load the data.
    vocab = load_vocab(vocab)

    # with open(questions_path) as f:
    #     questions = json.load(f)
    # it's so special if questions' number is 1, then will product error.
    questions=[{"question":" ".join(question.split("_")),"question_id":1,"answer":"zhan wei","image_id":image_path},
               {"question":" ".join(question.split("_")),"question_id":2,"answer":"zhan wei","image_id":image_path}]
    # Get the mappings from qid to answers.

    qid2ans, image_ids = create_questions_images_ids(questions)
    total_questions = len(qid2ans)
    total_images = len(image_ids)
    # print("Number of images to be written: %d" % total_images)
    # print("Number of QAs to be written: %d" % total_questions)

    h5file = h5py.File(output, "w")
    d_questions = h5file.create_dataset(
        "questions", (total_questions, max_q_length), dtype='i')
    d_indices = h5file.create_dataset(
        "image_indices", (total_questions,), dtype='i')
    d_images = h5file.create_dataset(
        "images", (total_images, im_size, im_size, 3), dtype='f')
    d_answers = h5file.create_dataset(
        "answers", (total_questions, max_a_length), dtype='i')

    # Create the transforms we want to apply to every image.
    transform = transforms.Compose([
        transforms.Resize((im_size, im_size))])

    # Iterate and save all the questions and images.
    bar = progressbar.ProgressBar(maxval=total_questions)
    bar.start()
    i_index = 0
    q_index = 0
    done_img2idx = {}
    images_ids = []
    for entry in questions:
        image_id = entry['image_id']
        images = []
        images.append(image_id)
        question_id = entry['question_id']
        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        if image_id not in done_img2idx:
            try:
                image = Image.open(image_id).convert('RGB')
            except IOError:
                image = Image.open(image_id).convert('RGB')
            image = transform(image)
            d_images[i_index, :, :, :] = np.array(image)
            done_img2idx[image_id] = i_index
            i_index += 1
        q, length = process_text(entry['question'], vocab,
                                 max_length=max_q_length)

        d_questions[q_index, :length] = q
        a, length = process_text(entry['answer'], vocab,
                                 max_length=max_a_length)
        d_answers[q_index, :length] = a

        d_indices[q_index] = done_img2idx[image_id]
        images.append(int(d_indices[q_index]))
        images_ids.append(images)
        # print(str(d_indices[q_index]))
        q_index += 1
        bar.update(q_index)
    h5file.close()

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
    w2v=Vectors(cache="/data2/entity/bhy/VQADEMO/model_code/nlm/.vector_cache",name=name)##cache='.vector_cache/wiki-PubMed-w2v.txt.pt.pt'
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
    # print ('='*80)
    # print ('GROUND TRUTH')
    # print (gts[:args.num_show])
    # print ('-'*80)
    # print ('PREDICTIONS')
    # print (preds[:args.num_show])
    # print ('='*80)
    # scores = nlge.compute_metrics(ref_list=[gts], hyp_list=preds)
    return  preds[0]
    #return gts, preds

def get_loader(dataset, transform, batch_size, sampler=None,
                   shuffle=False, num_workers=1, max_examples=None,
                   indices=None):
    """Returns torch.utils.data.DataLoader for custom dataset.

    Args:
        dataset: Location of annotations hdf5 file.
        transform: Transformations that should be applied to the images.
        batch_size: How many data points per batch.
        sampler: Instance of WeightedRandomSampler.
        shuffle: Boolean that decides if the data should be returned in a
            random order.
        num_workers: Number of threads to use.
        max_examples: Used for debugging. Assumes that we have a
            maximum number of training examples.
        indices: List of indices to use.

    Returns:
        A torch.utils.data.DataLoader for custom engagement dataset.
    """
    vqa = VQADataset(dataset, transform=transform, max_examples=max_examples,
                    indices=indices)
    data_loader = torch.utils.data.DataLoader(dataset=vqa,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader



def main(args):
    """Loads the model and then calls evaluate().

    Args:
        args: Instance of ArgumentParser.
    """

    # Load the arguments.
    model_dir = args.model_path
    params = Dict2Obj(json.load(
            open(os.path.join(model_dir, "args.json"), "r")))

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

    save_dataset(args.test_image, args.test_questions,'test', args.vocab_path,
                 args.test_output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length, max_c_length=args.max_c_length)


    # Build data loader
    # logging.info("Building data loader...")

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
    # logging.info("Building data loader...")
    data_loader = get_loader(args.test_output, transform,
                                 args.batch_size, shuffle=False,
                                 num_workers=args.num_workers,
                                 max_examples=args.max_examples,
                                 indices=None)
    # logging.info("Done")

    # Build the models
    # logging.info('Creating IQ model...')
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
    # logging.info("Done")

    # logging.info("Loading model.")
    vqa.load_state_dict(torch.load(os.path.join(args.model_path,'save.pkl')))

    # Setup GPUs.
    if torch.cuda.is_available():
        # logging.info("Using available GPU...")
        vqa.cuda()

    answer = evaluate(vqa, data_loader, vocab, args, params)
    #gts, preds = evaluate(vqg, data_loader, vocab, args, params)

    # Print and save the scores.
    print(answer)
    # with open(os.path.join(model_dir, args.results_path), 'w') as results_file:
    #     json.dump(scores, results_file)
    # with open(os.path.join(model_dir, args.preds_path), 'w') as preds_file:
    #     json.dump(preds, preds_file)
    # with open(os.path.join(model_dir, args.gts_path), 'w') as gts_file:
    #     json.dump(gts, gts_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Session parameters.
    parser.add_argument('--name', type=str, default='name'
                        ,help='Name for loading trained models')
    parser.add_argument('--model-path', type=str, default='weights/tf1/'
                        ,help='Path for loading trained models')
    parser.add_argument('--results-path', type=str, default='results.json',
                        help='Path for saving results.')
    parser.add_argument('--preds-path', type=str, default='preds.json',
                        help='Path for saving predictions.')
    parser.add_argument('--gts-path', type=str, default='gts.json',
                        help='Path for saving ground truth.')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=8)
    #parser.add_argument('--pin_memory', default=True)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max-examples', type=int, default=None,
                        help='When set, only evalutes that many data points.')
    parser.add_argument('--num-show', type=int, default=50,
                        help='Number of predictions to print.')
    parser.add_argument('--from-answer', type=str, default='true',
                        help='When set, only evalutes iq model with answers;'
                        ' otherwise it tests iq with answer types.')
    # parser.add_argument('--state', type=str, default='1',
    #                     help='Path for saving results.')
    # Data parameters.
    parser.add_argument('--dataset', type=str,
                        default='data/train_vqa_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val-dataset', type=str,
                        default='data/val_vqa_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--test-dataset', type=str,
                        default='data/test_vqa_dataset.hdf5',
                        help='Location of sampling weights for training set.')
    # parser.add_argument('--train-dataset-weights', type=str,
    #                     default='data/vqa_train_dataset_weights.json',
    #                     help='Location of sampling weights for training set.')
    # parser.add_argument('--val-dataset-weights', type=str,
    #                     default='data/vqa_val_dataset_weights.json',
    #                     help='Location of sampling weights for training set.')
    # parser.add_argument('--test-dataset-weights', type=str,
    #                     default='data/vqa_test_dataset_weights.json',
    #                     help='Location of sampling weights for training set.')

    # save dataset.py

    parser.add_argument('--test-image', type=str,
                        default="dataset/test/synpic53988.jpg",
                        help='directory for resized images')

    parser.add_argument('--test-questions', type=str,
                        default="what is the primary abnormality in this image?",
                        help='Path for train annotation file.')
    # Outputs.
    parser.add_argument('--test-output', type=str,
                        default='/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019/qa.hdf5',
                        help='directory for resized images.')

    parser.add_argument('--vocab-path', type=str,
                        default='data/vocab_vqa.json',
                        help='Path for saving vocabulary wrapper.')

    # Hyperparameters.
    parser.add_argument('--im_size', type=int, default=299,
                        help='Size of images.')
    parser.add_argument('--max-q-length', type=int, default=20,
                        help='maximum sequence length for questions.')
    parser.add_argument('--max-a-length', type=int, default=20,
                        help='maximum sequence length for answers.')
    parser.add_argument('--max-c-length', type=int, default=20,
                        help='maximum sequence length for answers.')

    args = parser.parse_args()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()
