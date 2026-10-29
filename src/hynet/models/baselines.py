import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from .hynet import DoubleConv, AttentionBlock
from .harmnet import HarmNet

class UNet(nn.Module):
    """U-Net built from the same DoubleConv / AttentionBlock modules as HY-Net, so that capacity and normalization are comparable.

    `attention=True` gives the Attention U-Net (additive attention gates on the skip connections).
    """
    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512), attention=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.downs = nn.ModuleList()
        curr = in_channels
        for f in features:
            self.downs.append(DoubleConv(curr, f))
            curr = f
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        up_in = features[-1] * 2
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(up_in, f, 2, 2))
            if attention:
                self.attentions.append(AttentionBlock(F_g=f, F_l=f, F_int=f // 2))
            self.dec_convs.append(DoubleConv(f * 2, f))
            up_in = f
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for idx, skip in enumerate(reversed(skips)):
            g = self.ups[idx](x)
            if g.shape != skip.shape:
                g = TF.resize(g, size=skip.shape[2:])
            if len(self.attentions) > 0:
                skip = self.attentions[idx](g=g, x=skip)
            x = self.dec_convs[idx](torch.cat((skip, g), dim=1))
        return self.final_conv(x)

class AttentionUNet(UNet):
    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__(in_channels, out_channels, features, attention=True)

class HarmUNet(nn.Module):
    """HarmNet -> single-stream U-Net on the harmonized image only (ablation: harmonization without the dual-stream Y-Net)."""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, device: torch.device, weights_path):
        super().__init__()
        self.harmnet = HarmNet(input_shape, hidden_units, input_shape).to(device)
        if weights_path is not None:
            self.harmnet.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        self.unet = UNet(input_shape, output_shape).to(device)

    def forward(self, x):
        return self.unet(self.harmnet(x))
