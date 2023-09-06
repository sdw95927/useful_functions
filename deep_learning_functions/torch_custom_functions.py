class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs) 
        inputs = torch.clamp(inputs, 0, 1)
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class MSELossCustom(nn.Module):
    def __init__(self, ignore_index=255, reduction="mean", gamma=0):
        super(MSELossCustom, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.gamma = gamma
        
    def forward(self, input, target):
        mask = target == self.ignore_index
        out = (input[~mask]-target[~mask])**2
        out_exp = torch.exp(out)
        if self.reduction == "mean":
            return out.mean()
            # return ((1 - out_exp) ** self.gamma * out).mean()
        elif self.reduction == "None":
            return out
            # return (1 - out_exp) ** self.gamma * out
