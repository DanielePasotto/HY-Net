import torch.nn as nn

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
