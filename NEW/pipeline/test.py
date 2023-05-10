from pathlib import Path
from typing import List
import logging

import torch
from torch import nn, Tensor
import torch.backends.cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from kornia.metrics import AverageMeter
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio import SignalDistortionRatio as SDR

from utils.CHiMe3 import CHiME3
from utils.environment_probe import EnvironmentProbe
from modules.generator_real import Generator


class Test:

    def __init__(self, config, path: Path = Path('/home/qiuzheng/Desktop/project/NEW/cache/best_complex_cotrain_15.4.pth'), cudnn: bool = True, half: bool = False, eval: bool = False, env: str = 'test'):
        torch.backends.cudnn.benchmark = cudnn
        self.environment_probe = EnvironmentProbe()
        self.device = self.environment_probe.device
        self.dict = torch.load(path, map_location=self.device)
        self.env = env

        net = Generator()
        net.load_state_dict(self.dict['g'])
        self.half = half
        _ = net.half() if half else None
        _ = net.cuda(1)
        _ = net.eval() if eval else None
        self.net = net

        dataset = CHiME3(config.folder, 'test',
                         config.n_sample, config.sr)
        self.dataloader = DataLoader(
            dataset, config.batch_size, True, num_workers=config.num_workers, pin_memory=True, drop_last=True)


    @torch.no_grad()
    def __call__(self):
        p_bar = tqdm(enumerate(self.dataloader), total=len(self.dataloader))
        meter = AverageMeter()
        for idx, sample in p_bar:
            sample = sample.to(self.environment_probe.device)
            c1, gt, c2 = torch.chunk(sample, 3, dim=1)

            vir = torch.squeeze(self.net(c1, c2))
            gt = torch.squeeze(gt)
            c1 = torch.squeeze(c1)

            r_loss = self.sdr(c1, gt).mean()
            v_loss = self.sdr(vir, gt).mean()

            p_bar.set_description(
                f'env: {self.env} SDR r_loss: {r_loss.item():03f} v_loss: {v_loss.item():03f}')
            meter.update(Tensor([r_loss, v_loss]))
        print(
            f'env: {self.env} SDR r_loss ave: {meter.avg[0]}, v_loss ave: {meter.avg[1]}')
