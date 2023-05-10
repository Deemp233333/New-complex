import torch
import torch.nn as nn
from torch import Tensor
from torchinfo import summary


class Generator(nn.Module):
    """
    Use two real singal spec to generate virtual sound.
    c1 + c2 -> vi
    """

    def __init__(self, depth: int = 9):
        super(Generator, self).__init__()
        self.depth = depth

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(2**(i+1), 2**(i+2), 16, 2, 7),
                    nn.BatchNorm1d(2**(i+2)),
                    nn.LeakyReLU(0.3)
                )for i in range(depth)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose1d(2**(depth-i+2), 2**(depth-i), 16, 2, 7),
                    nn.BatchNorm1d(2**(depth-i)),
                    nn.LeakyReLU(0.3)
                )for i in range(depth)
            ]
        )

        self.out = nn.Sequential(
            nn.Conv1d(2, 1, 15, 1, 7),
            nn.Tanh()
        )

        self.make_z = nn.Conv1d(2, 2**(depth+1), 512, 512, 0)

    def forward(self, c1: Tensor, c2: Tensor) -> Tensor:
        # Concat
        x = torch.cat([c1, c2], dim=1)

        # Make latent feature
        z = self.make_z(x)

        # forward
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        for layer in self.decoder:
            x = features.pop()
            z = layer(torch.cat((x, z), dim=1))

        return self.out(z)

    def pretrain(self, c1: Tensor, c2: Tensor, index: int):
        # Concat
        x = torch.cat([c1, c2], dim=1)

        # Make latent feature
        downsampling = nn.Conv1d(
            2, 2**(index+2), 2**(index+1), 2**(index+1), 0).cuda(3)
        z = downsampling(x)

        # forward
        features = []
        for i in range(index + 1):
            x = self.encoder[i](x)
            features.append(x)
        for i in range(index + 1):
            x = features.pop()
            z = self.decoder[self.depth-index +
                             i - 1](torch.cat((x, z), dim=1))

        return self.out(z)


if __name__ == '__main__':
    g = Generator()
    summary(g, [(16, 1, 64000), (16, 1, 64000)])
