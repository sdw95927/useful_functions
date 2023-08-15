class MSELossCustom(nn.Module):
    def __init__(self, ignore_index=255, reduction="mean"):
        super(MSELossCustom, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, input, target):
        mask = target == self.ignore_index
        out = (input[~mask]-target[~mask])**2
        if self.reduction == "mean":
            return out.mean()
        elif self.reduction == "None":
            return out
