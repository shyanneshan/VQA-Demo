"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa)
"""
import functools
import json
import operator
import os
import argparse
import _pickle as cPickle
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import dataset_RAD
import base_model
from train import train
import utils
from sklearn.metrics import accuracy_score, recall_score, f1_score

try:
    import _pickle as pickle
except:
    import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--name', type=str,default="model_epoch19.pth",
                        help='trained model name')
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models/san_mevf',
                        help='save file directory')

    # Utilities
    parser.add_argument('--seed', type=int, default=1204,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=20,
                        help='the number of epoches')
    parser.add_argument('--lr', default=0.005, type=float, metavar='lr',
                        help='initial learning rate')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--update_freq', default='1', metavar='N',
                        help='update parameters every n batches in an epoch')

    # Choices of attention models
    parser.add_argument('--model', type=str, default='BAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU','RNN'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - support testing, gpu training or sampling
    parser.add_argument('--print_interval', default=20, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--gpu', type=int, default=0,#bhy
                        help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')

    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
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
    # parser.add_argument('--use_RAD', action='store_true', default=True,#False
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
    parser.add_argument('--autoencoder', default=True,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='/data2/entity/bhy/VQADEMO/model_code/ODL/pretrained/pretrained_ae.pth',
                        help='the maml_model_path we use')
    parser.add_argument('--ae_alpha', default=0.001, type=float, metavar='ae_alpha',
                        help='ae_alpha')

    # MAML component hyper-parameters
    parser.add_argument('--maml', default=True,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='/data2/entity/bhy/VQADEMO/model_code/ODL/pretrained/pretrained_maml.weights',
                        help='the maml_model_path we use')

    # Return args
    args = parser.parse_args()
    return args
import tools.process2json
import tools.create_dictionary
import tools.create_embedding
import tools.compute_softscore

if __name__ == '__main__':
    args = parse_args()
    #process2json
    datasetpath = args.dataset

    tools.process2json.create_question_json(datasetpath)

    tools.process2json.mergeimgs(datasetpath)

    tools.process2json.create_img2val(datasetpath)

    tools.process2json.create_128(datasetpath)
    tools.process2json.create_84(datasetpath)
    #create dictionary
    # RAD_dir = '../data_RAD'
    d = tools.create_dictionary.create_dictionary(datasetpath)
    d.dump_to_file(datasetpath + '/dictionary.pkl')

    d = dataset_RAD.Dictionary.load_from_file(datasetpath + '/dictionary.pkl')
    emb_dim = 300
    glove_file = '/data2/entity/bhy/VQADEMO/model_code/ODL/glove/glove.6B.%dd.txt' % emb_dim
    weights, word2emb = tools.create_dictionary.create_glove_embedding_init(d.idx2word, glove_file)
    np.save(datasetpath + '/glove6b_init_%dd.npy' % emb_dim, weights)
    #compute_softscore
    train_answers = json.load(open(datasetpath + '/trainset.json'))

    answers = train_answers
    occurence = tools.compute_softscore.filter_answers(answers, 0)

    # print('occ', (len(occurence)))
    cache_path = datasetpath + '/cache/trainval_ans2label.pkl'
    if os.path.isfile(cache_path):
        # print('found %s' % cache_path)
        ans2label = cPickle.load(open(cache_path, 'rb'))
    else:
        # print("create")
        ans2label = tools.compute_softscore.create_ans2label(occurence, 'trainval',datasetpath + '/cache')

    tools.compute_softscore.compute_target(train_answers, ans2label, 'train',datasetpath + '/cache')

    val_answers = json.load(open( datasetpath + '/valset.json'))
    tools.compute_softscore.compute_target(val_answers, ans2label, 'val',datasetpath + '/cache')

    test_answers = json.load(open(datasetpath + '/testset.json'))
    tools.compute_softscore.compute_target(test_answers, ans2label, 'test',datasetpath + '/cache')


    #create_embedding
    emb_dims = [300]
    weights = [0] * len(emb_dims)
    # print(weights)
    if not os.path.exists(datasetpath + '/cache'):
        os.mkdir(datasetpath + '/cache')

    label2ans = cPickle.load(open(datasetpath + '/cache/trainval_label2ans.pkl', 'rb'))

    for idx, emb_dim in enumerate(emb_dims): # available embedding sizes
        glove_file = '/data2/entity/bhy/VQADEMO/model_code/ODL/glove/glove.6B.%dd.txt' % emb_dim
        weights[idx], word2emb = tools.create_dictionary.create_glove_embedding_init(label2ans, glove_file)
    np.save(datasetpath + '/glove6b_emb_%dd.npy' % functools.reduce(operator.add, emb_dims), np.hstack(weights))

    # create output directory and log file
    utils.create_dir(args.output)
    # logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    # logger.write(args.__repr__())
    # Set GPU device
    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device
    # Fixed ramdom seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    # Load dictionary and RAD training dataset
    # if args.use_RAD:
    #     dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.dataset, 'dictionary.pkl'))
    #     train_dset = dataset_RAD.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
    dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.dataset, 'dictionary.pkl'))
    train_dset = dataset_RAD.VQAFeatureDataset('train', args, dictionary, question_len=args.question_len)
    eval_dset = dataset_RAD.VQAFeatureDataset('val', args, dictionary, question_len=args.question_len)
    test_dset = dataset_RAD.VQAFeatureDataset('test', args, dictionary)
    batch_size = args.batch_size
    # Create VQA model
    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args)
    # print(model)
    optim = None
    epoch = 0
    # load snapshot
    if args.input is not None:
        # print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        model.to(device)
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1
    # create training dataloader
    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate, pin_memory=True)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate, pin_memory=True)
    test_loader = DataLoader(test_dset, batch_size, shuffle=False, num_workers=0, pin_memory=True,
                             collate_fn=utils.trim_collate)
    # training phase
    model_path=os.path.join(args.output,args.name+'.pth')
    train(args, model, train_loader, eval_loader, args.epochs, model_path, optim, epoch)


    def get_result(model, dataloader, device, args):
        with torch.no_grad():
            for v, q, a in iter(dataloader):  # , ans_type, q_types, p_type
                # if p_type[0] != "freeform":
                #     continue
                if args.maml:
                    v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
                if args.autoencoder:
                    v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
                v[0] = v[0].to(device)
                v[1] = v[1].to(device)
                q = q.to(device)
                a = a.to(device)
                # inference and get logit
                if args.autoencoder:
                    features, _ = model(v, q)
                else:
                    features = model(v, q)
                preds = model.classifier(features)
                final_preds = preds
                real=a.data.cpu()#.numpy()
                pred=final_preds.data.cpu()#.numpy()
                for i in range(len(pred)):
                    max_value = max(pred[i])
                    for j in range(len(pred[i])):
                        if max_value == pred[i][j]:
                            pred[i][j] = 1
                        else:
                            pred[i][j] = 0
                real=np.reshape(real,(-1))
                pred = np.reshape(pred, (-1))
                acc = accuracy_score(real, pred)
                recall = recall_score(real, pred)
                f1 = f1_score(real, pred)
                print(round(float(format(acc,'.4f'))*10000),round(float(format(recall,'.4f'))*10000),round(float(format(f1,'.4f'))*10000))
                # batch_score = compute_score_with_logits(final_preds, a.data).sum()


    # Testing process
    def process(args, model, eval_loader):
        model_path = os.path.join(args.output,args.name+'.pth')
        # print('loading %s' % model_path)
        model_data = torch.load(model_path)

        # Comment because do not use multi gpu
        # model = nn.DataParallel(model)
        model = model.to(args.device)
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        get_result(model, eval_loader, args.device, args)

        return

    process(args, model, test_loader)
