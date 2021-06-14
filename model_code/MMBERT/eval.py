import argparse
from utils import seed_everything, Model, VQAMed, test, load_data, LabelSmoothing
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torch.cuda.amp import GradScaler
# from torchtoolbox.transform import Cutout
# import os
# import pytorch_lightning as pl
import warnings

warnings.simplefilter("ignore", UserWarning)

def extract_one_data(question,imagpath):
    imag, ques, answ = [],[],[]
    # f = open(file,'r')
    # lines = f.readlines()
    # for line in lines:
    imag.append(imagpath)
    ques.append(question)
    answ.append(0)# test
    # f.close()
    temp={"img_id":imag,
          "question":ques,
          "answer":answ}
    df=pd.DataFrame(temp)
    return df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Evaluate")

    # parser.add_argument('--run_name', type = str, required = True, help = "run name for wandb")
    parser.add_argument('--run_name',default = "mmbert", type = str,help = "run name for wandb")
    parser.add_argument('--question', type=str,
                        default='what modality is shown?',
                        help='test txt(imag|question|answer)')
    parser.add_argument('--imag', type=str,
                        default='/home/wxl/Documents/VQADEMO/dataset/VQA-Med-2019/VQAMed2019Test/VQAMed2019_Test_Images/synpic54082.jpg',
                        help='test imag')
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
    parser.add_argument('--model_dir', type = str, required = False, default = "/home/wxl/Documents/VQADEMO/model_code/MMBERT/save/mmbert_acc.pt", help = "path to load weights")
    parser.add_argument('--save_dir', type = str, required = False, default = "/home/wxl/Documents/VQADEMO/model_code/MMBERT/save", help = "path to save weights")
    parser.add_argument('--category', type = str, required = False, default = None,  help = "choose specific category if you want")
    parser.add_argument('--use_pretrained', action = 'store_true', default = True, help = "use pretrained weights or not")
    parser.add_argument('--mixed_precision', action = 'store_true', default = False, help = "use mixed precision or not")
    parser.add_argument('--clip', action = 'store_true', default = False, help = "clip the gradients or not")

    parser.add_argument('--seed', type = int, required = False, default = 42, help = "set seed for reproducibility")
    parser.add_argument('--num_workers', type = int, required = False, default = 4, help = "number of workers")
    parser.add_argument('--epochs', type = int, required = False, default = 100, help = "num epochs to train")
    parser.add_argument('--train_pct', type = float, required = False, default = 1.0, help = "fraction of train samples to select")
    parser.add_argument('--valid_pct', type = float, required = False, default = 1.0, help = "fraction of validation samples to select")
    parser.add_argument('--test_pct', type = float, required = False, default = 1.0, help = "fraction of test samples to select")

    parser.add_argument('--max_position_embeddings', type = int, required = False, default = 28, help = "max length of sequence")
    parser.add_argument('--batch_size', type = int, required = False, default = 16, help = "batch size")
    parser.add_argument('--lr', type = float, required = False, default = 1e-4, help = "learning rate'")
    # parser.add_argument('--weight_decay', type = float, required = False, default = 1e-2, help = " weight decay for gradients")
    parser.add_argument('--factor', type = float, required = False, default = 0.1, help = "factor for rlp")
    parser.add_argument('--patience', type = int, required = False, default = 10, help = "patience for rlp")
    # parser.add_argument('--lr_min', type = float, required = False, default = 1e-6, help = "minimum lr for Cosine Annealing")
    parser.add_argument('--hidden_dropout_prob', type = float, required = False, default = 0.3, help = "hidden dropout probability")
    parser.add_argument('--smoothing', type = float, required = False, default = None, help = "label smoothing")

    parser.add_argument('--image_size', type = int, required = False, default = 224, help = "image size")
    '''hidden_size 312'''
    # 原先默认是312 但是会报错 就是img和txt连不上 不知道为啥 猜测是他用的模型不能很好的合上 是不是更新了？
    parser.add_argument('--hidden_size', type = int, required = False, default = 768, help = "hidden size")
    parser.add_argument('--vocab_size', type = int, required = False, default = 30522, help = "vocab size")
    parser.add_argument('--type_vocab_size', type = int, required = False, default = 2, help = "type vocab size")
    parser.add_argument('--heads', type = int, required = False, default = 12, help = "heads")
    parser.add_argument('--n_layers', type = int, required = False, default = 4, help = "num of layers")
    parser.add_argument('--num_vis', type = int, required = False,default=5, help = "num of visual embeddings")# 3 or 5 本来应该是True


    args = parser.parse_args()

    # wandb.init(project='medvqa', name = args.run_name, config = args)

    seed_everything(args.seed)


    train_df, val_df, test_df2 = load_data(args)
    test_df=extract_one_data(args.question,args.imag)
    import pre_work

    df= pre_work.getDfWithMode(args)


    ans2idx = {ans:idx for idx,ans in enumerate(df['answer'].unique())}
    idx2ans = {idx:ans for ans,idx in ans2idx.items()}




    # test_df['answer'] = test_df['answer'].map(ans2idx).astype(int)
    # train_df = df[df['mode']=='train'].reset_index(drop=True)
    # val_df = df[df['mode']=='val'].reset_index(drop=True)
    # test_df = test_df[df['mode']=='test'].reset_index(drop=True)

    num_classes = len(ans2idx)

    args.num_classes = num_classes

    # train_df = pd.concat([train_df, val_df]).reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model(args)

    model.classifier[2] = nn.Linear(args.hidden_size, num_classes)

    if args.use_pretrained:
        model.load_state_dict(torch.load(args.model_dir))

        
    model.to(device)

    # wandb.watch(model, log='all')


    optimizer = optim.Adam(model.parameters(),lr=args.lr)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = args.patience, factor = args.factor, verbose = True)


    if args.smoothing:
        criterion = LabelSmoothing(smoothing=args.smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    scaler = GradScaler()


    test_tfm = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



    testdataset = VQAMed(test_df, imgsize = args.image_size, tfm = test_tfm, args = args)

    testloader = DataLoader(testdataset, batch_size = 1, shuffle=False, num_workers = args.num_workers)

    best_acc1 = 0
    best_acc2 = 0
    best_loss = np.inf
    counter = 0

    test_loss, predictions, acc, bleu ,_,_,_= test(testloader, model, criterion, device, scaler, args, test_df,idx2ans)
    # model = abn_yn_model1()
    # model.load_state_dict(load(path))
 
    test_df['preds'] = predictions
    test_df['decode_preds'] = test_df['preds'].map(idx2ans)
    test_df['decode_ans'] = test_df['answer'].map(idx2ans)
    print(test_df['decode_preds'].values[0])
    # test_df.to_csv(f'test_csvs/{args.category}_test_mlm_preds.csv', index = False)


            