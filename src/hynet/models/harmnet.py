import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super(ResNetBlock, self).__init__()

        self.skip = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=output_shape, kernel_size=1),
            nn.InstanceNorm2d(output_shape, affine=True)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=output_shape, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(output_shape, affine=True),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=output_shape, out_channels=output_shape, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(output_shape, affine=True),
            nn.LeakyReLU()
        )

    def forward(self, x):
        identity = self.skip(x)
        x = self.conv_block(x)
        x += identity
        return x

class HarmNet(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super(HarmNet, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(hidden_units, affine=True),
            nn.LeakyReLU()
        )

        self.down = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(hidden_units * 2, affine=True),
            nn.LeakyReLU()
        )

        self.res_blocks = nn.Sequential(*[ResNetBlock(hidden_units * 2, hidden_units * 2) for _ in range(4)])

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_units * 2, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(hidden_units, affine=True),
            nn.LeakyReLU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(hidden_units, affine=True),
            nn.LeakyReLU()
        )

        self.out = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=output_shape, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_1(x)
        residual = x
        x = self.down(x)
        x = self.res_blocks(x)
        x = self.up(x)
        x = self.conv_2(x)
        x += residual
        x = self.out(x)
        return x

class ResidualHarmNet(HarmNet):
    """HarmNet that predicts a bounded correction of its input: out = x + scale * tanh(conv(features)).

    The last convolution starts at zero, so the network is the identity at initialization and only changes the image
    where the segmentation loss asks for it. It needs no pre-training on a classical-pipeline target.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, scale: float = 0.5) -> None:
        assert input_shape == output_shape
        super().__init__(input_shape, hidden_units, output_shape)
        self.scale = scale
        conv = self.out[0]
        nn.init.zeros_(conv.weight)
        nn.init.zeros_(conv.bias)
        self.out = nn.Sequential(conv, nn.Tanh())

    def forward(self, x):
        return x + self.scale * super().forward(x)


class ParametricHarmNet(nn.Module):
    """Constrained harmonizer: it can only apply a smooth illumination correction and a global monotone tone curve, so it cannot draw structures.

    A small network looks at a 96x128 version of the image and predicts, on a coarse grid, a log-gain and an offset (bilinearly interpolated to the image size,
    so the field has no detail finer than a few tens of pixels) and the increments of a monotone piecewise-linear tone curve. The corrected image is standardized
    per image (mean 0.5, std 0.25, clipped to [0, 1]) before the curve, so the output cannot collapse to a constant. The last layers start at zero: at
    initialization the output is the standardized input.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, grid=(12, 16), knots: int = 8, max_log_gain: float = 1.0, max_offset: float = 0.5) -> None:
        assert input_shape == output_shape == 1
        super().__init__()
        c = max(hidden_units // 4, 8)
        layers, ch = [], input_shape
        for _ in range(3):  # 96x128 -> 12x16
            layers += [nn.Conv2d(ch, c, 3, stride=2, padding=1), nn.InstanceNorm2d(c, affine=True), nn.ReLU(inplace=True)]
            ch = c
        self.encoder = nn.Sequential(*layers)
        self.field = nn.Conv2d(c, 2, 1)
        self.tone = nn.Linear(c, knots)
        for m in (self.field, self.tone):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.knots, self.max_log_gain, self.max_offset = knots, max_log_gain, max_offset

    def forward(self, x):
        feats = self.encoder(F.interpolate(x, size=(96, 128), mode="area"))
        field = F.interpolate(self.field(feats), size=x.shape[2:], mode="bilinear", align_corners=False)
        z = x * torch.exp(self.max_log_gain * torch.tanh(field[:, :1])) + self.max_offset * torch.tanh(field[:, 1:])
        mean, std = z.mean(dim=(1, 2, 3), keepdim=True), z.std(dim=(1, 2, 3), keepdim=True)
        u = ((z - mean) / (4 * std + 1e-6) + 0.5).clamp(0, 1)
        inc = torch.softmax(self.tone(feats.mean(dim=(2, 3))), dim=1)[:, :, None, None]  # (B, K, 1, 1), sums to 1
        steps = torch.arange(self.knots, device=x.device, dtype=x.dtype)[None, :, None, None]
        return (inc * (u * self.knots - steps).clamp(0, 1)).sum(dim=1, keepdim=True)
