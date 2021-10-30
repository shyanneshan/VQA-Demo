"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
import argparse
import torch
from torch.utils.data import DataLoader
import dataset_RAD
import base_model
import utils
import pandas as pd
import os
import json
# answer_types = ['CLOSED', 'OPEN', 'ALL']
# quesntion_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble flag. If True, generate a logit file which is used in the ensemble part')
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--question', type=str, default='what organ system is displayed in this ct scan?')
    parser.add_argument('--imag', type=str, default='data_RAD/test/synpic16333.jpg')
    parser.add_argument('--input', type=str, default='saved_models/san_mevf/model_epoch19.pth',
                        help='input file directory for loading a model')
    # Utilities
    # parser.add_argument('--epoch', type=int, default=19,
    #                     help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    # Choices of Attention models
    parser.add_argument('--model', type=str, default='SAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['RNN','LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - gpu
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    # Question embedding
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Train with RAD
    # parser.add_argument('--use_RAD', action='store_true', default=False,
    #                     help='Using TDIUC dataset to train')
    parser.add_argument('--dataset', type=str,default='data_RAD',
                        help='RAD dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=64, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder',  default=True,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='/data2/entity/bhy/VQADEMO/model_code/ODL/pretrained/pretrained_ae.pth',
                        help='the maml_model_path we use')

    # MAML component hyper-parameters
    parser.add_argument('--maml', default=True,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='/data2/entity/bhy/VQADEMO/model_code/ODL/pretrained/pretrained_maml.weights',
                        help='the maml_model_path we use')

    # Return args
    args = parser.parse_args()
    return args
# Load questions
def get_question(q, dataloader):
    q = q.squeeze(0)
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)

# Load answers
def get_answer(p, dataset):
    _m, idx = p.max(1)
    return dataset.label2ans[idx.item()]

# Logit computation (for train, test or evaluate)
def get_result(model, dataset, device, args):
    qa_res={}
    keys = ['count', 'real', 'true', 'real_percent', 'score', 'score_percent']
    # question_types_result = dict((i, dict((j, dict((k, 0.0) for k in keys)) for j in quesntion_types)) for i in answer_types)
    # result = dict((i, dict((j, 0.0) for j in keys)) for i in answer_types)
    with torch.no_grad():
        v=[0,0]
        q=dataset.question.unsqueeze(0)
        if args.maml:
            v[0] = dataset.maml_images_data.reshape(dataset.maml_images_data.shape[0], 84, 84).unsqueeze(1)
        if args.autoencoder:
            v[1] = dataset.ae_images_data.reshape(dataset.ae_images_data.shape[0], 128, 128).unsqueeze(1)
        v[0] = v[0].to(device)
        v[1] = v[1].to(device)
        q = q.to(device)
        # v = v.to(device)
        # inference and get logit
        if args.autoencoder:
            features, _ = model(v, q)
        else:
            features = model(v, q)
        preds = model.classifier(features)
        final_preds = preds
        s_answer = get_answer(final_preds, dataset)
        # for v, q in dataset:#, ans_type, q_types, p_type
            # if args.maml:
            #     v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
            # if args.autoencoder:
            #     v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
            # v[0] = v[0].to(device)
            # v[1] = v[1].to(device)
            # q = q.to(device)
            # # inference and get logit
            # if args.autoencoder:
            #     features, _ = model(v, q)
            # else:
            #     features = model(v, q)
            # preds = model.classifier(features)
            # final_preds = preds
            # s_answer = get_answer(final_preds, dataloader)

    return s_answer #, question_types_result

# Test phase
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble flag. If True, generate a logit file which is used in the ensemble part')
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--question', type=str, default='what organ system is displayed in this ct scan?')
    parser.add_argument('--imag', type=str, default='data_RAD/test/synpic16333.jpg')
    parser.add_argument('--input', type=str, default='saved_models/san_mevf/model_epoch19.pth',
                        help='input file directory for loading a model')
    # Utilities
    # parser.add_argument('--epoch', type=int, default=19,
    #                     help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size')

    # Choices of Attention models
    parser.add_argument('--model', type=str, default='SAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['RNN', 'LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - gpu
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    # Question embedding
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Train with RAD
    # parser.add_argument('--use_RAD', action='store_true', default=False,
    #                     help='Using TDIUC dataset to train')
    parser.add_argument('--dataset', type=str, default='data_RAD',
                        help='RAD dir')

    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=64, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', default=True,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str,
                        default='/data2/entity/bhy/VQADEMO/model_code/ODL/pretrained/pretrained_ae.pth',
                        help='the maml_model_path we use')

    # MAML component hyper-parameters
    parser.add_argument('--maml', default=True,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str,
                        default='/data2/entity/bhy/VQADEMO/model_code/ODL/pretrained/pretrained_maml.weights',
                        help='the maml_model_path we use')

    # Return args
    args = parser.parse_args()
    # args = parse_args()
    # # print(args)
    # print("hello")
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

    # Check if evaluating on TDIUC dataset or VQA dataset
    # if args.use_RAD:
    #     dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir , 'dictionary.pkl'))
    #     eval_dset = dataset_RAD.VQAFeatureDataset(args.split, args, dictionary)


    dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.dataset, 'dictionary.pkl'))

    eval_dset = dataset_RAD.VQAPredict(args, dictionary)

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args)
    # print(model)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=utils.trim_collate)

    def save_questiontype_results(outfile_path, quesntion_types_result):
        for i in quesntion_types_result:
            pd.DataFrame(quesntion_types_result[i]).transpose().to_csv(outfile_path + '/question_type_' + i + '.csv')
    # Testing process
    def process(args, model, eval_loader):

        model_path = args.input # + '/model_epoch%s.pth' % args.epoch
        # print('loading %s' % model_path)
        model_data = torch.load(model_path)

        model = model.to(args.device)
        model.load_state_dict(model_data.get('model_state', model_data))

        answer = get_result(model, eval_dset, args.device, args)

        print(answer)

        return
    process(args, model, eval_loader)
