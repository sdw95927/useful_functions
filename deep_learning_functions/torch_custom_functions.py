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
            # return ((1 - out_exp) ** self.gamma * out).mean()  # Remove "1 - "?
        elif self.reduction == "None":
            return out
            # return (1 - out_exp) ** self.gamma * out

  
class CustomeInstanceNorm2d(nn.Module):
    def __init__(self, ngf):
        super(CustomeInstanceNorm2d, self).__init__()
        self.channels = ngf
    def forward(self, x):
        # https://nn.labml.ai/normalization/instance_norm/index.html
        '''
        mean = torch.mean(x, dim=(0, 1))
        var = torch.var(x, dim=(0, 1), unbiased=False)
        return (x - mean)/torch.sqrt(var ** 2 + 1e-5)
        '''
        '''
        x_shape = x.shape
        batch_size = x_shape[0]
        x = x.view(batch_size, self.channels, -1)
        mean = x.mean(dim=[-1], keepdim=True)
        mean_x2 = (x ** 2).mean(dim=[-1], keepdim=True)
        var = mean_x2 - mean ** 2
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        x_norm = x_norm.view(batch_size, self.channels, -1)
        return x_norm.view(x_shape)
        '''
        mean = x.mean(dim=(2, 3), keepdim=True)
        mean_x2 = (x ** 2).mean(dim=(2, 3), keepdim=True)
        var = mean_x2 - mean ** 2
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return x_norm
        

def get_norm_layer(norm_type):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = CustomeInstanceNorm2d
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
