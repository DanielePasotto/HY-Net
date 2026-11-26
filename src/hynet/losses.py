import torch.nn as nn
import torch.nn.functional as F

class TverskyBCELoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, smooth=1):
        super(TverskyBCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        return BCE + (1 - Tversky)
