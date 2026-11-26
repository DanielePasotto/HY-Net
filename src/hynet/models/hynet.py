import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from .harmnet import ResidualHarmNet

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
             g1 = TF.resize(g1, size=x1.shape[2:])
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class GatedFusionBlock(nn.Module):
    def __init__(self, channels):
        super(GatedFusionBlock, self).__init__()
        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        if x1.shape != x2.shape:
             x2 = TF.resize(x2, size=x1.shape[2:])

        cat = torch.cat((x1, x2), dim=1)
        alpha = self.gate_conv(cat)

        return (x1 * alpha) + (x2 * (1 - alpha))

class AsymmetricGatedYNet(nn.Module):
    """Dual-encoder Y-Net. The defaults are the published model; `use_attention=False, fusion="concat"` is the ablation that feeds the raw skips
    unfiltered and concatenates the two streams instead of the learned pixel-wise gate."""
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], use_attention=True, fusion="gated"):
        super(AsymmetricGatedYNet, self).__init__()
        assert fusion in ("gated", "concat")
        self.use_attention = use_attention
        self.gated = fusion == "gated"
        self.fusion = fusion
        streams = 2 if fusion == "concat" else 1  # channel multiplier of the fused features

        self.encoder1_layers = nn.ModuleList()
        self.encoder2_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        curr = in_channels
        for f in features:
            self.encoder1_layers.append(DoubleConv(curr, f))
            self.encoder2_layers.append(DoubleConv(curr, f))
            curr = f

        if self.gated:
            self.bottleneck_fusion = GatedFusionBlock(features[-1])
        self.bottleneck = DoubleConv(features[-1] * streams, features[-1] * 2)

        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        self.dec_convs = nn.ModuleList()

        reversed_features = features[::-1]
        bottleneck_out = features[-1] * 2

        for f in reversed_features:
            self.ups.append(nn.ConvTranspose2d(bottleneck_out, f, 2, 2))
            if use_attention:
                self.attentions.append(AttentionBlock(F_g=f, F_l=f, F_int=f//2))
            if self.gated:
                self.fusion_blocks.append(GatedFusionBlock(f))
            self.dec_convs.append(DoubleConv(f * streams + f, f))
            bottleneck_out = f

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x1, x2):
        skips1 = []
        skips2 = []

        for layer in self.encoder1_layers:
            x1 = layer(x1); skips1.append(x1); x1 = self.pool(x1)

        for layer in self.encoder2_layers:
            x2 = layer(x2); skips2.append(x2); x2 = self.pool(x2)

        if self.gated:
            fused_bn = self.bottleneck_fusion(x1, x2)
        else:
            fused_bn = torch.cat((x1, x2), dim=1)
        x = self.bottleneck(fused_bn)
        skips1 = skips1[::-1]
        skips2 = skips2[::-1]

        for idx in range(len(self.ups)):
            g = self.ups[idx](x)
            s1 = skips1[idx]
            s2 = skips2[idx]

            if g.shape != s1.shape: g = TF.resize(g, size=s1.shape[2:])

            s2_filtered = self.attentions[idx](g=g, x=s2) if self.use_attention else s2
            if self.gated:
                fused_skip = self.fusion_blocks[idx](s1, s2_filtered)
            else:
                fused_skip = torch.cat((s1, s2_filtered), dim=1)
            concat = torch.cat((fused_skip, g), dim=1)
            x = self.dec_convs[idx](concat)

        return self.final_conv(x)

class HYNet(nn.Module):
    """HarmNet followed by the Asymmetric Gated Y-Net (the harmonized and the raw image feed the two streams).

    HarmNet is a ResidualHarmNet (identity at initialization, trained only through the segmentation loss). `harmonize=False` skips it and gives
    the raw image to both streams (Y-Net without HarmNet). `harm_down` > 1 makes HarmNet work on a reduced image (faster inference).
    `use_attention`/`fusion` are forwarded to the Y-Net.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, device: torch.device,
                 harmonize: bool = True, use_attention: bool = True, fusion: str = "gated", harm_down: int = 1) -> None:
        super(HYNet, self).__init__()
        self.harmonize = harmonize
        if harmonize:
            self.harmnet = ResidualHarmNet(input_shape, hidden_units, input_shape, down=harm_down).to(device)
        self.ynet = AsymmetricGatedYNet(input_shape, output_shape, use_attention=use_attention, fusion=fusion).to(device)

    def forward_proc(self, x):
        """(logits, image that feeds the first stream); the second is None without HarmNet."""
        if not self.harmonize:
            return self.ynet(x, x), None
        proc = self.harmnet(x)
        return self.ynet(proc, x), proc

    def forward(self, x):
        return self.forward_proc(x)[0]
