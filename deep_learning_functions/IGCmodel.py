"""Adapted from https://github.com/HobbitLong/CMC/blob/master/NCE/NCEAverage.py
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

from alias_multinomial import AliasMethod

class NCEAverage(nn.Module):

    def __init__(self, input_size, output_size, K, T=0.07, momentum=0.5, use_softmax=False, 
                 device="cuda: 0"):
        """
        Args:
            input_size: n_features
            output_size: n_samples
            K: number of negatives to constrast for each positive
            T: temperature that modulates the distribution
        """
        super(NCEAverage, self).__init__()
        self.output_size = output_size
        self.unigrams = torch.ones(self.output_size)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.to(device)
        self.K = K
        self.use_softmax = use_softmax

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(input_size / 3)
        self.register_buffer('memory_image', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
        self.register_buffer('memory_gene', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))

    def forward(self, image, gene, index, idx=None):
        """
        Args:
            image: out_features for image
            gene: out_features for gene
            index: torch.long for data ids
        """
        K = int(self.params[0].item())
        T = self.params[1].item()
        Z_image = self.params[2].item()  # Normalization constant for image output
        Z_gene = self.params[3].item()  # Normalization constant for gene output

        momentum = self.params[4].item()
        batch_size = image.size(0)
        output_size = self.memory_image.size(0)
        input_size = self.memory_image.size(1)

        # score computation
        if idx is None:
            idx = self.multinomial.draw(batch_size * (self.K + 1)).view(batch_size, -1)
            idx.select(1, 0).copy_(index.data)
        # sample
        weight_gene = torch.index_select(self.memory_gene, 0, idx.view(-1)).detach()
        weight_gene = weight_gene.view(batch_size, K + 1, input_size)
        out_image = torch.bmm(weight_gene, image.view(batch_size, input_size, 1))
        # sample
        weight_image = torch.index_select(self.memory_image, 0, idx.view(-1)).detach()
        weight_image = weight_image.view(batch_size, K + 1, input_size)
        # Batchwise matrix multiplication
        # weight_image: [batch_size, K + 1, n_out_features]
        # gene: [batch_size, n_out_features]
        # out_gene:[batch_size, K + 1, 1]
        out_gene = torch.bmm(weight_image, gene.view(batch_size, input_size, 1))

        if self.use_softmax:
            out_image = torch.div(out_image, T)
            out_gene = torch.div(out_gene, T)
            out_image = out_image.contiguous()
            out_gene = out_gene.contiguous()
        else:
            out_image_e = torch.exp(out_image - torch.max(out_image, dim=1, keepdim=True)[0])
            out_image_s = torch.sum(out_image_e, dim=1, keepdim=True)
            out_image = torch.div(out_image_e, out_image_s)
            
            out_gene_e = torch.exp(out_gene - torch.max(out_gene, dim=1, keepdim=True)[0])
            out_gene_s = torch.sum(out_gene_e, dim=1, keepdim=True)
            out_gene = torch.div(out_gene_e, out_gene_s)
            """
            out_image = torch.exp(torch.div(out_image, T))
            out_gene = torch.exp(torch.div(out_gene, T))
            # set Z_0 if haven't been set yet,
            # Z_0 is used as a constant approximation of Z, to scale the probs
            if Z_image < 0:
               self.params[2] = out_image.mean() * output_size
                Z_image = self.params[2].clone().detach().item()
                print("normalization constant Z_image is set to {:.1f}".format(Z_image))
            if Z_gene < 0:
                self.params[3] = out_gene.mean() * output_size
                Z_gene = self.params[3].clone().detach().item()
                print("normalization constant Z_gene is set to {:.1f}".format(Z_gene))
            # compute out_image, out_gene
            out_image = torch.div(out_image, Z_image).contiguous()
            out_gene = torch.div(out_gene, Z_gene).contiguous()
            """

        # # update memory
        with torch.no_grad():
            image_pos = torch.index_select(self.memory_image, 0, index.view(-1))
            image_pos.mul_(momentum)
            image_pos.add_(torch.mul(image, 1 - momentum))
            image_norm = image_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_image = image_pos.div(image_norm)
            self.memory_image.index_copy_(0, index, updated_image)

            gene_pos = torch.index_select(self.memory_gene, 0, index.view(-1))
            gene_pos.mul_(momentum)
            gene_pos.add_(torch.mul(gene, 1 - momentum))
            gene_norm = gene_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_gene = gene_pos.div(gene_norm)
            self.memory_gene.index_copy_(0, index, updated_gene)

        return out_image, out_gene

class IGCNet(torch.nn.Module):
    def __init__(self, image_shape, n_genes, n_out_features, 
                 imagenet='resnet101', 
                 genenet=[100, 100]):
        """
        Args:
            image_shape: w/h of input image
            n_genes: number of genes to consider
            n_out_features: number of output features
            imagenet: architecture to encode image features
            genenet: architecture to encode genetic features: list of #hidden nodes
        """
        super(IGCNet, self).__init__()
        
        if imagenet == 'resnet101':
            # Require image size at least 224
            self.imagenet = torch.hub.load('pytorch/vision:v0.4.0', imagenet, pretrained=True)
            self.imagenet.float()
            self.imagenet.layer4[2].relu = torch.nn.ReLU6()
            self.imagenet.fc = torch.nn.Linear(self.imagenet.fc.in_features, n_out_features)
            input_size = image_shape
        elif imagenet == 'inception_v3':
            # Require image input size = 299, has auxillary output
            self.imagenet = torch.hub.load('pytorch/vision:v0.4.0', imagenet, pretrained=True)
            self.imagenet.float()
            raise("Incomplete model")
        else:
            raise("Invalid model name")
        
        genenet_fcs = []
        genenet_in_shape = n_genes
        for i, genenet_n_hidden_nodes in enumerate(genenet):
            genenet_fcs.append(torch.nn.Linear(genenet_in_shape, genenet_n_hidden_nodes))
            genenet_fcs.append(torch.nn.BatchNorm1d(genenet_n_hidden_nodes))
            genenet_fcs.append(torch.nn.ReLU6())
            genenet_in_shape = genenet_n_hidden_nodes
        genenet_fcs.append(torch.nn.Linear(genenet_in_shape, n_out_features))
        self.genenet_fcs = torch.nn.ModuleList(genenet_fcs)
        
    def forward(self, data):
        """
        Args:
            data: a dictionary
        """
        image, f_gene = data['image'], data['gene']
        f_image = self.imagenet(image)
        for i, genenet_fc in enumerate(self.genenet_fcs):
            if i < len(self.genenet_fcs) - 1:
                # f_gene = F.relu(genenet_fc(f_gene))
                f_gene = genenet_fc(f_gene)
            else:
                f_gene = genenet_fc(f_gene)
        return f_image, f_gene