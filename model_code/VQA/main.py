import argparse
import os

import torch

from utils import Vocabulary
from utils.vocab import create_vocab_api
from utils.store_dataset import create_dataset_all
from train_vqa import train
from evaluate_vqa import main



if __name__=="__main__":
    import sys
    sys.path.append('/home/wxl/Documents/VQADEMO/model_code/VQA/utils')
    # sys.path.append('/home/wxl/Documents/VQADEMO/model_code/VQA/data')
    # print(os.path.dirname(__file__))
    parser = argparse.ArgumentParser()
    #project location should be config must!
    parser.add_argument('--project_path', type=str,
                        default=os.path.dirname(__file__),
                        help='train txt(imag|question|answer)')
    #create vocab
    parser.add_argument('--train_text_file', type=str,
                        default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt',
                        help='train txt(imag|question|answer)')
    parser.add_argument('--valid_text_file', type=str,
                        default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt',
                        help='valid txt(imag|question|answer)')
    parser.add_argument('--test_text_file', type=str,
                        default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.txt',
                        help='test txt(imag|question|answer)')
    parser.add_argument('--train_image_file', type=str,
                        default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Training/Train_images/',
                        help='train imag')
    parser.add_argument('--valid_image_file', type=str,
                        default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/ImageClef-2019-VQA-Med-Validation/Val_images/',
                        help='valid imag')
    parser.add_argument('--test_image_file', type=str,
                        default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/VQAMed2019Test/VQAMed2019_Test_Images/',
                        help='test imag')

    # Session parameters.
    parser.add_argument('--run_name', type=str, default='name',
                        help='train name and model name')

    parser.add_argument('--model_path', type=str, default='weights/tf1/',
                        help='Path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='Size for randomly cropping images')
    parser.add_argument('--log_step', type=int, default=10,
                        help='Step size for prining log info')
    parser.add_argument('--save_step', type=int, default=None,
                        help='Step size for saving trained models')
    parser.add_argument('--eval_steps', type=int, default=100,
                        help='Number of eval steps to run.')
    parser.add_argument('--eval_every_n_steps', type=int, default=100,#400,
                        help='Run eval after every N steps.')
    parser.add_argument('--num_epochs', type=int, default=1)#20
    parser.add_argument('--batch_size', type=int, default=16)#32 nei cun hui zha diao! [chinese
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--info_learning_rate', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_examples', type=int, default=None,
                        help='For debugging. Limit examples in database.')

    # Lambda values.
    parser.add_argument('--lambda_gen', type=float, default=1.0,
                        help='coefficient to be added in front of the generation loss.')
    parser.add_argument('--lambda_z', type=float, default=0.001,
                        help='coefficient to be added in front of the kl loss.')
    parser.add_argument('--lambda_t', type=float, default=0.0001,
                        help='coefficient to be added with the type space loss.')
    parser.add_argument('--lambda_a', type=float, default=0.001,
                        help='coefficient to be added with the answer recon loss.')
    parser.add_argument('--lambda_i', type=float, default=0.001,
                        help='coefficient to be added with the image recon loss.')
    parser.add_argument('--lambda_z_t', type=float, default=0.001,
                        help='coefficient to be added with the t and z space loss.')
    # args = parser.parse_args()
    # Data parameters.
    # parser.add_argument('--vocab-path', type=str,
    #                     default='data/vqamed1/vocab_vqa.json',
    #                     help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,required=True,
                        # default='data/'+args.run_name+'/vqa_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--val_dataset', type=str,required=True,
                        # default='data/'+args.run_name+'/vqa_dataset_val.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--train_dataset_weights', type=str,required=True,
                        # default='data/'+args.run_name+'/vqa_train_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--val_dataset_weights', type=str,required=True,
                        # default='data/'+args.run_name+'/vqa_test_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Location of where the model weights are.')


    # Model parameters
    parser.add_argument('--rnn_cell', type=str, default='LSTM',
                        help='Type of rnn cell (GRU, RNN or LSTM).')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers in lstm.')
    parser.add_argument('--max_length', type=int, default=20,
                        help='Maximum sequence length for outputs.')
    parser.add_argument('--encoder_max_len', type=int, default=20,
                        help='Maximum sequence length for inputs.')
    parser.add_argument('--bidirectional', action='store_true', default=False,
                        help='Boolean whether the RNN is bidirectional.')
    parser.add_argument('--use_glove',  type=str,default=False,#action='store_true',
                        help='Whether to use GloVe embeddings.')
    parser.add_argument('--use_w2v', type=str,default=False,#action='store_true',
                        help='Whether to use W2V embeddings.')
    parser.add_argument('--embedding_name', type=str, default='840B',#PubMed-w2v.txt
                        help='Name of the GloVe embedding to use. data/processed/PubMed-w2v.txt')
    parser.add_argument('--num_categories', type=int, default=5,
                        help='Number of answer types we use.')
    parser.add_argument('--dropout_p', type=float, default=0.3,
                        help='Dropout applied to the RNN model.')
    parser.add_argument('--input_dropout_p', type=float, default=0.3,
                        help='Dropout applied to inputs of the RNN.')
    parser.add_argument('--num_att_layers', type=int, default=2,
                        help='Number of attention layers.')
    parser.add_argument('--z_size', type=int, default=100,
                        help='Dimensions to use for hidden variational space.')

    # Ablations.
    parser.add_argument('--no_image_recon', action='store_true', default=True,
                        help='Does not try to reconstruct image.')
    parser.add_argument('--no_question_recon', action='store_true', default=True,
                        help='Does not try to reconstruct answer.')
    parser.add_argument('--no_caption_recon', action='store_true', default=False,
                        help='Does not try to reconstruct answer.')
    parser.add_argument('--no_category_space', action='store_true', default=False,
                        help='Does not try to reconstruct answer.')

    # args = parser.parse_args()
    parser.add_argument('--vocab_path', required=True,
                        # default='data/'+args.run_name+'/vocab_vqa.json',
                        help='Does not try to reconstruct answer.')

    #eval
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--num_show', type=int, default=50,
                        help='Number of predictions to print.')
    parser.add_argument('--from_answer', type=str, default='true',
                        help='When set, only evalutes iq model with answers;'
                        ' otherwise it tests iq with answer types.')
    # args.model_path + "vqa-tf-" + args.state + ".pkl"
    # Data parameters.
    # parser.add_argument('--dataset', type=str,
    #                     default='data/'+args.run_name+'/vqa_dataset_val.hdf5',
    #                     help='path for train annotation json file')
    # parser.add_argument('--state', type=str, required=True,
    #                     # default=str(args.num_epochs),
    #                     help='version of saving results.')
    args = parser.parse_args()
    # vocab_path='data/'+args.run_name+'/vocab_vqa.json'
    create_vocab_api(args.project_path,args.run_name,args.train_text_file,
                     args.valid_text_file,args.test_text_file,4,args.vocab_path)
    Vocabulary()
    output = os.path.join(args.model_path,args.run_name)
    im_size = 299
    max_q_length = 20
    max_a_length = 20
    max_c_length = 20
    # print(output)
    create_dataset_all(args.train_image_file, args.train_text_file, args.valid_image_file,
                       args.valid_text_file, args.project_path+'/'+args.vocab_path, output, im_size, max_q_length,
                       max_a_length, max_c_length)
    Vocabulary()

    model_list=train(args)
    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()


    # args = parser.parse_args()
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
    # Hack to disable errors for importing Vocabulary. Ignore this line.
    Vocabulary()