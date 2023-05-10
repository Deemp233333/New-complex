from pathlib import Path
from typing import List

import torch
from torch import nn, Tensor
import torch.backends.cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from kornia.metrics import AverageMeter
from torchmetrics import SignalDistortionRatio as SDR

from utils.CHiMe3 import CHiME3
from utils.environment_probe import EnvironmentProbe


class Eval:

    def __init__(self, net, config, cudnn: bool = True, half: bool = False, eval: bool = False):
        torch.backends.cudnn.benchmark = cudnn

        self.environment_probe = EnvironmentProbe()

        self.device = self.environment_probe.device
        self.half = half
        _ = net.half() if half else None
        _ = net.cuda(self.device)
        _ = net.eval() if eval else None
        self.net = net

        dataset = CHiME3(config.folder, 'valid',
                         config.n_sample, config.sr)
        self.dataloader = DataLoader(
            dataset, config.batch_size, True, num_workers=config.num_workers, pin_memory=True, drop_last=True)

        self.l1 = nn.L1Loss(reduction='none')
        self.l1.cuda(self.device)

        self.sdr = SDR()
        self.sdr.cuda(self.device)

    @torch.no_grad()
    def __call__(self):
        p_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        meter = AverageMeter()
        for idx, sample in p_bar:
            sample = sample.to(self.device)
            c1, gt, c2 = torch.chunk(sample, 3, dim=1)

            vir = self.net(c1, c2)

            loss = self.l1(vir, gt).mean()
            sdr = self.sdr(vir, gt).mean()

            p_bar.set_description(
                f'L1 loss: {loss.item():03f} | SDR: {sdr.item():03f}')
            meter.update(Tensor([loss, sdr]))
        print(f'L1 loss ave: {meter.avg[0]} | SDR ave: {meter.avg[1]}')
        return meter.avg[1].item()
