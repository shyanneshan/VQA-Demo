# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:57:57 2020
@author: Xuandi Fu(Carnegie Mellon University)
"""
import torch
# from resnet import resnet as caffe_resnet
import numpy as np
from torchvision import models
import torch.nn as nn

def myLinear(input_len, output_len,batch_size):
    return nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.BatchNorm1d(batch_size),#2
            nn.ReLU()
        )


def outputLinear(input_len, output_len):
    return nn.Sequential(
            nn.Linear(input_len, output_len),
            nn.Sigmoid()
    )

class ArticleNet(nn.Module):
    def __init__(self,args):
        super(ArticleNet, self).__init__()
        self.imag_model = models.resnet50(pretrained=True)
        self.gru_q = nn.GRU(args.q_len, args.q_hidden, args.q_layer)
        self.gru_title = nn.GRU(args.tit_len, args.tit_hidden, args.t_layer)
        self.gru_sent = nn.GRU(args.sent_len, args.sent_hidden, args.s_layer)
        # batch_size能够堆满
        self.fc_title = myLinear(args.tit_hidden, args.art_hidden,args.batch_size)
        self.fc_sent = myLinear(args.sent_hidden, args.art_hidden,args.batch_size)
        # batch_size堆不满
        # self.fc_title_2 = myLinear2(config['tit_hidden'], config['art_hidden'], config['batch_size'])
        # self.fc_sent_2 = myLinear2(config['sent_hidden'], config['art_hidden'], config['batch_size'])
        # self.title_class = outputLinear(config['art_hidden'], config['num_class'])  # score num_class = 1
        # self.sent_class = outputLinear(config['art_hidden'], config['num_class'])
        # self.art_class = outputLinear(config['art_hidden'], config['num_class'])
        self.title_class = outputLinear(args.art_hidden, args.title_num_class) # score num_class = 1
        self.sent_class = outputLinear(args.art_hidden, args.sent_num_class)
        self.art_class = outputLinear(args.art_hidden, args.art_num_class)

    def forward(self, question, imag, title, sents, length,args):

        #image: the  visual features Vtaken from an ImageNet trained ResNet152
        x = self.imag_model.conv1(imag.to(torch.float64))
        x = self.imag_model.bn1(x)
        x = self.imag_model.relu(x)
        x = self.imag_model.maxpool(x)
        x = self.imag_model.layer1(x)
        x = self.imag_model.layer2(x)
        x = self.imag_model.layer3(x)
        x = self.imag_model.layer4(x)
        image = self.imag_model.avgpool(x)
        image=image.squeeze(2).squeeze(2).unsqueeze(0)
        # final_image=np.zeros((1,1,128))
        final_image=[]
        for idx in range(128):
            final_image.append(image[:,:,(idx+1)*16-1].detach().numpy())
            # final_image[:,:,idx]=image[:,:,(idx+1)*16-1].detach().numpy()
        # final_image = torch.from_numpy(final_image)
        final_image=torch.tensor(final_image).transpose(0,1).transpose(1,2)
        #question: question featureg_model(imag)
        question = question.unsqueeze(0)
        title = title.unsqueeze(0)
        sents= sents.transpose(0,1)
        _, hq = self.gru_q(question)
        hqv = hq + final_image

        atitle, h_title = self.gru_title(title, hqv)
        #hq = hq.repeat(100, 1, 1)
        asents, h_sent = self.gru_sent(sents, hqv)
        # h=h_title.squeeze()

        if question.shape[1]<args.batch_size:
            fc_title_2 = myLinear(args.tit_hidden, args.art_hidden, question.shape[1])
            fc_sent_2 = myLinear(args.sent_hidden, args.art_hidden, question.shape[1])
            title_emb = fc_title_2(h_title.to(torch.float32))
            sent_emb = fc_sent_2(h_sent.to(torch.float32))
        else:
            title_emb = self.fc_title(h_title)
            sent_emb = self.fc_sent(h_sent)

        # h_title=torch.from_numpy(np.zeros((128,128)))
        # h_sent=torch.from_numpy(np.zeros((128, 128)))
        # title_emb = self.fc_title(h_title)
        # sent_emb = self.fc_sent(h_sent)

        hart = title_emb + sent_emb
        atitle = self.title_class(h_title + hart).squeeze(0)
        aart = self.art_class(hart.to(torch.float64)).squeeze(0)
        asents = self.sent_class(h_sent + hart).squeeze(0)
        # ret=torch.hstack((atitle, aart, asents))
        # return torch.vstack((atitle, aart, asents))
        return atitle,asents,aart

