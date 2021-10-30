import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
# import wandb
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import seed_everything, Model, VQAMed, test, load_data, LabelSmoothing, predict, validate

warnings.simplefilter("ignore", UserWarning)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = "Evaluate")

    # parser.add_argument('--run_name', type = str, required = True, help = "run name for wandb")
    parser.add_argument('--question', type = str, required = False, default = "what is the primary abnormality in this image?", help = "predict question")
    parser.add_argument('--imag', type = str, required = False, default = "/data2/entity/bhy/VQADEMO/uploadimag/7uSwcTru7kb8tos.jpg", help = "predict image path")
    parser.add_argument('--data_dir', type = str, required = False, default = "/data2/entity/bhy/VQADEMO/dataset/VQA-Med-2019", help = "path for data")
    parser.add_argument('--model_dir', type = str, required = False, default = "/data2/entity/bhy/VQADEMO/weights/1028mm/1028mm.pt", help = "path to load weights")
    parser.add_argument('--save_dir', type = str, required = False, default = "/home/viraj.bagal/viraj/medvqa/Weights/ic19", help = "path to save weights")
    parser.add_argument('--category', type = str, required = False, default = None,  help = "choose specific category if you want")
    parser.add_argument('--use_pretrained', action = 'store_true', default = False, help = "use pretrained weights or not")
    parser.add_argument('--mixed_precision', action = 'store_true', default = False, help = "use mixed precision or not")
    parser.add_argument('--clip', action = 'store_true', default = False, help = "clip the gradients or not")

    parser.add_argument('--seed', type = int, required = False, default = 42, help = "set seed for reproducibility")
    parser.add_argument('--num_workers', type = int, required = False, default = 4, help = "number of workers")
    # parser.add_argument('--epochs', type = int, required = False, default = 100, help = "num epochs to train")
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
    parser.add_argument('--hidden_size', type = int, required = False, default = 768, help = "hidden size")
    parser.add_argument('--vocab_size', type = int, required = False, default = 30522, help = "vocab size")
    parser.add_argument('--type_vocab_size', type = int, required = False, default = 2, help = "type vocab size")
    parser.add_argument('--heads', type = int, required = False, default = 12, help = "heads")
    parser.add_argument('--n_layers', type = int, required = False, default = 4, help = "num of layers")
    parser.add_argument('--num_vis', type=int, required=False, help="num of visual embeddings")


    args = parser.parse_args()

    # wandb.init(project='medvqa', name = args.run_name, config = args)

    seed_everything(args.seed)


    train_df, val_df, test_df = load_data(args)


    if args.category:
            
        train_df = train_df[train_df['category']==args.category].reset_index(drop=True)
        val_df = val_df[val_df['category']==args.category].reset_index(drop=True)
        test_df = test_df[test_df['category']==args.category].reset_index(drop=True)


    df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    ans2idx = {ans:idx for idx,ans in enumerate(df['answer'].unique())}
    idx2ans = {idx:ans for ans,idx in ans2idx.items()}




    df['answer'] = df['answer'].map(ans2idx).astype(int)
    train_df = df[df['mode']=='train'].reset_index(drop=True)
    val_df = df[df['mode']=='val'].reset_index(drop=True)
    test_df = df[df['mode']=='test'].reset_index(drop=True)

    predict_df = pd.DataFrame(columns = ["img_id","question","answer","mode"])
    new_dataframe = predict_df.append({"img_id":args.imag,"question":" ".join(args.question.split("_")),
                                       "answer":0,"mode":"test"}, ignore_index = True)

    num_classes = len(ans2idx)

    args.num_classes = num_classes

    train_df = pd.concat([train_df, val_df]).reset_index(drop=True)

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


    # test_tfm = transforms.Compose([transforms.ToTensor(),
    #                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_tfm = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    testdataset = VQAMed(new_dataframe, imgsize = args.image_size, tfm = test_tfm, args = args)
    # testdataset = VQAMed(test_df, imgsize = args.image_size, tfm = test_tfm, args = args)

    testloader = DataLoader(testdataset, batch_size = args.batch_size, shuffle=False, num_workers = args.num_workers)

    best_acc1 = 0
    best_acc2 = 0
    best_loss = np.inf
    counter = 0

    val_loss, PREDS, acc, bleu = predict(testloader,model,criterion,device,scaler,args,test_df,idx2ans)

    print(idx2ans[PREDS[0]])
    # test_df['preds'] = PREDS
    # test_df['decode_preds'] = test_df['preds'].map(idx2ans)#!!!i need it.
    # test_df['decode_ans'] = test_df['answer'].map(idx2ans)
    # test_df.to_csv('test_mlm_preds.csv', index = False)


            