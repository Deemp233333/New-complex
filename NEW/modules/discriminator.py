import torch
from torch import nn, Tensor
from torch.nn.utils.parametrizations import spectral_norm as sn
from torchinfo import summary


class Discriminator(nn.Module):
    """
    Use to discriminate grandtruth signal and signal generated by the generator
    """

    def __init__(self, depth: int = 4):
        super(Discriminator, self).__init__()

        self.conv = nn.ModuleList(
            [
                nn.Sequential(
                    sn(nn.Conv1d(2**(i), 2**(i+1), 16, 4, 6)),
                    nn.BatchNorm1d(2**(i+1)),
                    nn.LeakyReLU(0.3)
                )for i in range(depth)
            ]
        )

        self.flatten = nn.Flatten()

        self.linear = nn.Sequential(
            sn(nn.Linear(4000, 256)),
            sn(nn.Linear(256, 1)),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.conv:
            x = layer(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    d = Discriminator()
    summary(d, (16, 1, 64000))
