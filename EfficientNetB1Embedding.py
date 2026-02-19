import torch
import torch.nn as nn
import timm

class EfficientNetB1Embedding   (nn.Module):
    def __init__(self, pretrained=True):
        super(EfficientNetB1Embedding, self).__init__()
        # Load backbone
        self.backbone = timm.create_model('efficientnet_b1', pretrained=pretrained, num_classes = 0)
        self.embedding_dim = self.backbone.num_features
    def forward(self, x):
        return self.backbone(x)
