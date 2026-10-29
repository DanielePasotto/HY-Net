import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM

class TVLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = self.tensor_size(x[:, :, 1:, :])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()

        count_w = self.tensor_size(x[:, :, :, 1:])
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()

        return self.weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        k_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        kernel_x = torch.FloatTensor(k_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(k_y).unsqueeze(0).unsqueeze(0)

        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)

        target_grad_x = F.conv2d(target, self.kernel_x, padding=1)
        target_grad_y = F.conv2d(target, self.kernel_y, padding=1)

        pred_mag = torch.abs(pred_grad_x) + torch.abs(pred_grad_y)
        target_mag = torch.abs(target_grad_x) + torch.abs(target_grad_y)

        return self.l1_loss(pred_mag, target_mag)

class RegionAwareCompositeLoss(nn.Module):
    def __init__(self, lambda_mae=1.0, lambda_ssim=0.5, lambda_tv=0.1, lambda_edge=0.1, lambda_bg=0.5):
        super(RegionAwareCompositeLoss, self).__init__()
        self.w_mae = lambda_mae
        self.w_ssim = lambda_ssim
        self.w_tv = lambda_tv
        self.w_edge = lambda_edge
        self.w_bg = lambda_bg

        self.l1 = nn.L1Loss()
        self.edge = EdgeLoss()
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=1)
        self.tv = TVLoss(weight=1.0)

        self.blur = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.blur.weight.data.fill_(1.0 / 25.0)
        self.blur.weight.requires_grad = False

    def forward(self, pred, target, mask, input_img):

        loss_mae = self.l1(pred * mask, target * mask)
        loss_edge = self.edge(pred * mask, target * mask)
        loss_ssim = 1 - self.ssim(pred * mask, target * mask)

        loss_tv = self.tv(pred)

        bg_mask = 1 - mask

        with torch.no_grad():
            bg_target = self.blur(input_img)

        loss_bg = self.l1(pred * bg_mask, bg_target * bg_mask)

        total = (self.w_mae * loss_mae) + \
                (self.w_edge * loss_edge) + \
                (self.w_ssim * loss_ssim) + \
                (self.w_tv * loss_tv) + \
                (self.w_bg * loss_bg)

        return total, loss_ssim.item()

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
