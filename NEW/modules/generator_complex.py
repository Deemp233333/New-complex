import torch
import torch.nn as nn
from torch import Tensor
from modules.complex_layers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU, ComplexTanh
from torchinfo import summary


class Generator(nn.Module):
    """
    Use two real singal spec to generate virtual sound.
    c1 + c2 -> vi
    """

    def __init__(self, depth: int = 5):
        super(Generator, self).__init__()
        self.depth = depth

        self.encoder = nn.ModuleList([
            nn.Sequential(
                ComplexConv2d(2 ** (i+1), 2**(i+2), kernel_size=(3, 3),
                              stride=1, dilation=1, padding=(1, 1)),
                ComplexBatchNorm2d(2**(i+2)),
                ComplexReLU()
            ) for i in range(depth)
        ])

        self.decoder = nn.ModuleList([
            nn.Sequential(
                ComplexConv2d(dim * (i + 1), dim, kernel_size=(3, 3),
                              stride=1, dilation=1, padding=(1, 1)),
                ComplexBatchNorm2d(dim),
                ComplexReLU()
            ) for i in range(depth)
        ])

        self.generate = nn.Sequential(
            nn.Sequential(
                ComplexConv2d(dim * (depth + 1), dim * 4,
                              kernel_size=(3, 3), stride=1, padding=(1, 1)),
                ComplexReLU()
            ),
            nn.Sequential(
                ComplexConv2d(dim * 4, dim * 2, kernel_size=(3, 3),
                              stride=1, padding=(1, 1)),
                ComplexBatchNorm2d(dim * 2),
                ComplexReLU()
            ),
            nn.Sequential(
                ComplexConv2d(dim * 2, dim, kernel_size=(3, 3),
                              stride=1, padding=(1, 1)),
                ComplexBatchNorm2d(dim),
                ComplexReLU()
            ),
            nn.Sequential(
                ComplexConv2d(dim, 2, kernel_size=(3, 3),
                              stride=1, padding=(1, 1)),
                nn.Tanh()
            ),
        )

    def forward(self, c1: Tensor, c2: Tensor) -> Tensor:
        ori = torch.stack([c1, c2], dim=1)
        mask = torch.full_like(ori, 0.5*1.6487212707)
        x = self.encoder(mask)
        for i in range(self.depth):
            t = self.dense[i](x)
            x = torch.cat([x, t], dim=1)
        x = self.generate(x)
        vir = torch.mul(x, ori)
        ph, amp = torch.angle(vir), torch.abs(vir)
        ph = torch.sum(ph, dim=1)
        amp = torch.sum(amp, dim=1)
        vir = torch.polar(amp, ph)
        return vir
