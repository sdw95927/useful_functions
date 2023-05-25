"""Predict gene expression
"""

import torch
from torch import nn
import torch.nn.functional as F

class GENet(torch.nn.Module):
    def __init__(self, image_shape, n_genes, n_out_features, fix_imagenet=True, imagenet='resnet101'):
        """
        Predict gene expression with image.
        
        Args:
            image_shape: w/h of input image
            n_genes: number of genes to consider
            n_out_features: number of features right before last layer (equal to IGCNet)
            fix_imagenet: whether update image net weights during training
            imagenet: architecture to encode image features
        """
        super(GENet, self).__init__()
        
        if imagenet == 'resnet101':
            # Require image size at least 224
            self.imagenet = torch.hub.load('pytorch/vision:v0.4.0', imagenet, pretrained=True)
            self.imagenet.float()
            self.imagenet.fc = torch.nn.Linear(self.imagenet.fc.in_features, n_out_features)
            input_size = image_shape
        elif imagenet == 'inception_v3':
            # Require image input size = 299, has auxillary output
            self.imagenet = torch.hub.load('pytorch/vision:v0.4.0', imagenet, pretrained=True)
            self.imagenet.float()
            raise("Incomplete model")
        else:
            raise("Invalid model name")
        
        self.last_fc = torch.nn.Linear(n_out_features, n_genes)
        
        if fix_imagenet:
            for parameters in self.imagenet.parameters():
                parameters.requires_grad_(False)
        
    def forward(self, data):
        """
        Args:
            data: a dictionary
        """
        image = data['image']
        f_image = self.imagenet(image)
        ge = self.last_fc(f_image)
        return ge