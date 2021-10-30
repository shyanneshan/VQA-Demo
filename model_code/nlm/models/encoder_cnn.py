"""Genearates a representation for an image input.
"""

import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Generates a representation for an image input.
    """
    '''
    def __init__(self,embed_size):
        """Load the pretrained ResNet-50 and replace top fc layer."""

        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-2]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        #self.avgpool = nn.AvgPool2d(7)
        self.avgpool = nn.MaxPool2d(7)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.
	"""
        self.embed.weight.data.normal_(0.0, 0.02)
        self.embed.bias.data.fill_(0)
    def forward(self, images):
        """Extract feature vectors from input images."""
        # with torch.no_grad():
        #     features = self.resnet(images)
        # features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(features))

        features = self.resnet(images)  # need gradient
        features = self.avgpool(features)
        features = features.view(features.size(0), -1)  # reshape
        features = self.batch(self.embed(features))
        return features'''


    '''
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        #batch normalization
        self.batch= nn.BatchNorm1d(embed_size,momentum = 0.01)
        #Weights initialization
        self.embed.weight.data.normal_(0., 0.02)
        self.embed.bias.data.fill_(0)
        
    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.batch(self.embed(features))
        return features
    '''
  

    '''
    def __init__(self, output_size):
        """Load the pretrained ResNet-152 and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        self.cnn = models.densenet161(pretrained=True)#50
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.classifier = nn.Linear(self.cnn.classifier.in_features, output_size)
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights.
	"""
        self.cnn.classifier.weight.data.normal_(0.0, 0.02)
        self.cnn.classifier.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors.
	"""
        features = self.cnn(images)
        features = features.view(features.size(0), -1)
        output = self.bn(features)
        return output'''


    
    def __init__(self, output_size):

        super(EncoderCNN, self).__init__()
        self.cnn = models.resnet50(pretrained=True)#resnet18
        #self.cnn.aux_logits=False
        for param in self.cnn.parameters():
            param.requires_grad = True
            
        #num_ftrs = self.cnn.AuxLogits.fc.in_features
        #self.cnn.AuxLogits.fc = nn.Linear(num_ftrs, output_size)
        # Handle the primary net
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_ftrs,output_size)
        
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()
        """
        super(EncoderCNN, self).__init__()
        self.cnn = models.googlenet(pretrained=True)#resnet18
        for param in self.cnn.parameters():
            param.requires_grad = False
        num_features = self.cnn.classifier[6].in_features
        features = list(self.cnn.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, 512)])
        self.cnn.classifier=nn.Sequential(*features)
        #self.cnn.fc=nn.Sequential(*features)

        self.cnn.fc = nn.Linear(512, output_size)
        #self.cnn.classifier = nn.Sequential(*features)
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()"""
    def init_weights(self):
        """Initialize the weights.
	"""
        self.cnn.fc.weight.data.normal_(0.0, 0.02)
        self.cnn.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors.
	"""
        features = self.cnn(images)
        output = self.bn(features)
        return output


    