"""Transform all the VQA dataset into a hdf5 dataset.
"""

from PIL import Image
from torchvision import transforms

import argparse
import json
import h5py
import numpy as np
import os
import progressbar

# from utils.train_utils import Vocabulary
from utils import Vocabulary
from utils.vocab import load_vocab
from utils.vocab import process_text


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


def save_dataset(image_dir, questions_path, vocab, output,
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
    from vocab import getOneTxt
    questions=getOneTxt(questions_path)
    # with open(questions_path,'r',encoding='utf-8') as f:
    #     questions = json.load(f)

    # Get the mappings from qid to answers.
    
    qid2ans, image_ids = create_questions_images_ids(questions)
    total_questions = len(qid2ans)
    total_images = len(image_ids)
    # print ("Number of images to be written: %d" % total_images)
    # print ("Number of QAs to be written: %d" % total_questions)

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
    images_ids=[]
    for entry in questions:
        image_id = entry['image_id']
        images=[]
        images.append(image_id)
        question_id = entry['question_id']
        if image_id not in image_ids:
            continue
        if question_id not in qid2ans:
            continue
        if image_id not in done_img2idx:
            try:
                path = image_id
                image = Image.open(os.path.join(image_dir, path+".jpg")).convert('RGB')
            except IOError:
                path = image_id
                image = Image.open(os.path.join(image_dir, path)).convert('RGB')
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
        #print(str(d_indices[q_index]))
        q_index += 1
        bar.update(q_index)
    h5file.close()
    # print ("Number of images written: %d" % i_index)
    # print ("Number of QAs written: %d" % q_index)
    
    # with open(os.path.join('../data/vqamed1/', 'results_image_v.json'), 'w',encoding='utf-8') as results_file:
    #     json.dump(images_ids, results_file)


def create_dataset_all(train_image_dir,train_questions,val_image_dir,val_questions,
                       vocab_path,output,im_size,
                       max_q_length,max_a_length,max_c_length):
    #train
    save_dataset(train_image_dir, train_questions, vocab_path,
                 output+'/vqa_dataset.hdf5', im_size=im_size,
                 max_q_length=max_q_length, max_a_length=max_a_length, max_c_length=max_c_length)
    # print('Wrote dataset to %s/vqa_dataset.hdf5' % output)
    #val
    save_dataset(val_image_dir, val_questions, vocab_path,
                 output+'/vqa_dataset_val.hdf5', im_size=im_size,
                 max_q_length=max_q_length, max_a_length=max_a_length, max_c_length=max_c_length)
    # print('Wrote dataset to %s/vqa_dataset_val.hdf5' % output)
    # Hack to avoid import errors.
    Vocabulary()

def create_dataset_predict(image_dir,questions,vocab_path,output):
    #test
    im_size=299
    max_q_length=20
    max_a_length=20
    max_c_length=20
    save_dataset(image_dir, questions, vocab_path,
                 output+'/pred_dataset_val.hdf5', im_size=im_size,
                 max_q_length=max_q_length, max_a_length=max_a_length, max_c_length=max_c_length)
    # print('Wrote dataset to %s/pred_dataset_val.hdf5' % output)
    # Hack to avoid import errors.
    Vocabulary()

# if __name__ == '__main__':
#     #test
#     # train_image_dir='../data/vqamed1/predict'
#     # train_questions='../data/vqamed1/predict.txt'
#
#     train_image_dir='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Training/Train_images'
#     val_image_dir='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Validation/Val_images'
#     train_questions='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt'
#     val_questions='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt'
#     vocab_path='/home/wxl/Documents/VQADEMO/model_code/VQA/data/vqamed1/vocab_vqa.json'
#     output='/home/wxl/Documents/VQADEMO/model_code/VQA/data/vqamed1'
#     im_size=299
#     max_q_length=20
#     max_a_length=20
#     max_c_length=20
#     create_dataset_all(train_image_dir,train_questions,val_image_dir,
#                         val_questions,vocab_path,output,im_size,max_q_length,
#                        max_a_length,max_c_length)
