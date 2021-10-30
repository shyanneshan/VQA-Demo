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


def save_dataset(args,image_dir, questions_path, choice,vocab, output,
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

    with open(questions_path) as f:
        questions = json.load(f)

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
    
    with open(os.path.join(args.data_dir, choice+'_results_image_v.json'), 'w') as results_file:
        json.dump(images_ids, results_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Inputs. /home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/VQA-MeD-CELF/data_processing/ImageClef-2020/VQA-Med-2020-Task2-VQGeneration-TrainVal-Sets/VQAMed2020-VQGeneration-TrainingSet/VQGeneration_2020_Train_images/
    # /home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/VQA-MeD-CELF/data_processing/ImageClef-2020/VQA-Med-2020-Task2-VQGeneration-TrainVal-Sets/VQAMed2020-VQGeneration-ValidationSet/VQGeneration_2020_Validation_images/
    # /home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/VQA-MeD-CELF/data_processing/ImageClef-2020/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-ValidationSet/VQAnswering_2020_Val_images
    # /home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/VQA-MeD-CELF/data_processing/ImageClef-2020/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-TrainingSet/VQAnswering_2020_Train_images
    # /home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/VQA-MeD-CELF/data_processing/ImageClef-2020/VQA-Med-2020-Task1-VQAnswering-TestSet/VQA-Med-2020-Task1-VQAnswering-TestSet
    
    
    #/home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/VQA-MeD-CELF/data_processing/ImageClef-2020/VQA-Med-2020-Task1-VQAnswering-TestSet/VQA-Med-2020-Task1-VQAnswering-TestSet/Task1-2020-VQAnswering-Test-Images
    
    #/home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/VQA-MeD-CELF/data_processing/ImageClef-2020/vqa-vqg/Train_images
    
    #parser.add_argument('--image-dir', type=str, default='/home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/VQA-MeD-CELF/data_processing/ImageClef-2020/VQA-Med-2020-Task1-VQAnswering-TestSet/VQA-Med-2020-Task1-VQAnswering-TestSet/Task1-2020-VQAnswering-Test-Images',                        
     #                   help='directory for resized images')
    #parser.add_argument('--image-dir', type=str, default='/home/sarroutim2/PosDoc NLM/VQA/Codes/IQ/VQA-MeD-CELF/data_processing/ImageClef-2020/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQA-Med-2020-Task1-VQAnswering-TrainVal-Sets/VQAMed2020-VQAnswering-ValidationSet/VQAnswering_2020_Val_images/',                        
     #                   help='directory for resized images')
    parser.add_argument('--train-image-dir', type=str,
                        default='../dataset/train',
                        help='directory for resized images')

    parser.add_argument('--train-questions', type=str,
                        default='../dataset/train.json',
                        help='Path for train annotation file.')
    # Outputs.
    parser.add_argument('--train-output', type=str,
                        default='../data/train_vqa_dataset.hdf5',
                        help='directory for resized images.')

    parser.add_argument('--val-image-dir', type=str,
                        default='../dataset/val',
                        help='directory for resized images')

    parser.add_argument('--val-questions', type=str,
                        default='../dataset/val.json',
                        help='Path for train annotation file.')
    # Outputs.
    parser.add_argument('--val-output', type=str,
                        default='../data/val_vqa_dataset.hdf5',
                        help='directory for resized images.')

    parser.add_argument('--test-image-dir', type=str,
                        default='../dataset/test',
                        help='directory for resized images')   
    
    parser.add_argument('--test-questions', type=str,
                        default='../dataset/test.json',
                        help='Path for train annotation file.')
    # Outputs.
    parser.add_argument('--test-output', type=str,
                        default='../data/test_vqa_dataset.hdf5',
                        help='directory for resized images.')


    parser.add_argument('--vocab-path', type=str,
                        default='../data/vocab_vqa.json',
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

    save_dataset(args.train_image_dir, args.train_questions,'train', args.vocab_path,
                 args.train_output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length, max_c_length=args.max_c_length)
    # print('Train: Wrote dataset to %s' % args.train_output)
    # Hack to avoid import errors.
    Vocabulary()

    save_dataset(args.val_image_dir, args.val_questions, 'val',args.vocab_path,
                 args.val_output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length, max_c_length=args.max_c_length)
    # print('Val: Wrote dataset to %s' % args.val_output)

    save_dataset(args.test_image_dir, args.test_questions,'test', args.vocab_path,
                 args.test_output, im_size=args.im_size,
                 max_q_length=args.max_q_length, max_a_length=args.max_a_length, max_c_length=args.max_c_length)
    # print('Test: Wrote dataset to %s' % args.test_output)
