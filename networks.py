import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

# from efficientnet_pytorch import EfficientNet


class CMNIST_MLP(nn.Module):
    def __init__(self, input_shape):
        super(CMNIST_MLP, self).__init__()
        self.hdim = hdim = 390
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1] * input_shape[2], hdim),
            nn.ReLU(True),
            nn.Linear(hdim, hdim),
            nn.ReLU(True)
        )
        
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                gain = nn.init.calculate_gain('relu')
                nn.init.xavier_uniform_(m.weight, gain=gain)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)
    
    
class Classifier(nn.Module):
    def __init__(self, backbone, output_dim):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.hdim, output_dim)
    
    def forward(self, x):
        embs = self.backbone(x)
        embs = torch.reshape(embs, (x.size(0), -1))
        return self.fc(embs)
    
    
class EnvClassifier(nn.Module):
    def __init__(self, backbone, class_dim, logits_dim, output_dim):
        super(EnvClassifier, self).__init__()
        self.class_dim = class_dim
        self.logits_dim = logits_dim
        self.output_dim = output_dim
        self.backbone = backbone
        
        if isinstance(backbone, CMNIST_MLP):
            self.fc = nn.Linear(backbone.hdim, logits_dim)
            self.f = nn.Sequential(backbone, self.fc)
        elif isinstance(backbone, models.resnet.ResNet):
            setattr(backbone, 'hdim', backbone.fc.in_features)
            self.fc = backbone.fc = nn.Linear(backbone.hdim, logits_dim)
            self.f = backbone
        elif isinstance(backbone, EfficientNet):
            setattr(backbone, 'hdim', backbone._fc.in_features)
            self.fc = backbone._fc = nn.Linear(backbone.hdim, logits_dim)
            self.f = backbone
            
        self.g = nn.Linear(class_dim, logits_dim)
        self.h = nn.Linear(logits_dim, output_dim)
        
        for m in (self.fc, self.g, self.h):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x, y):
        if isinstance(self.backbone, CMNIST_MLP):
            x = x.view(x.size(0), -1)
        embs = self.f(x)
        weights = self.g(y)
        return embs, self.h(embs * weights)

    
class MMDClassifier(nn.Module):
    def __init__(self, backbone):
        super(MMDClassifier, self).__init__()
        self.backbone = backbone
        self.fc = nn.Linear(backbone.hdim, 1)
        
    def forward(self, x):
        embs = self.backbone(x)
        embs = torch.reshape(embs, (x.size(0), -1))
        return self.fc(embs)