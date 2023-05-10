from utils.CHiMe3 import CHiME3
from utils.environment_probe import EnvironmentProbe
from modules.generator_real import Generator
from modules.discriminator import Discriminator
from functions.div_loss import div_loss
from pipeline.eval import Eval

import logging
import pathlib
from functools import reduce
from pathlib import Path

import torch

from torch import nn, Tensor
from torch.optim import RMSprop, Adam
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from kornia.metrics import AverageMeter


class Train:
    """
    The train process for NEW
    """

    def __init__(self, environment_probe: EnvironmentProbe, config: dict):
        logging.info(
            f'NEW Training | weight: {config.weight}')
        self.config = config
        self.environment_probe = environment_probe

        # modules
        logging.info(f'generator | depth: {config.g_depth}')
        self.generator = Generator(config.g_depth)
        logging.info(
            f'discriminator | depth: {config.d_depth}')
        self.discriminator = Discriminator(config.d_depth)

        # WGAN adam optim
        logging.info(
            f'optimizer: {config.optimizer} | learning rate: {config.learning_rate}')

        if config.optimizer == 'RMSprop':
            self.opt_generator = RMSprop(
                self.generator.parameters(), lr=config.learning_rate)
            self.opt_discriminator = RMSprop(
                self.discriminator.parameters(), lr=config.learning_rate)

        else:
            self.opt_generator = Adam(
                self.generator.parameters(), lr=config.learning_rate, betas=(0.9, 0.95))
            self.opt_discriminator = Adam(
                self.discriminator.parameters(), lr=config.learning_rate)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt_generator, 'max')
        # move to device
        logging.info(f'module device: {environment_probe.device}')
        self.generator.to(environment_probe.device)
        self.discriminator.to(environment_probe.device)

        # loss
        self.l1 = nn.L1Loss(reduction='none')
        self.l1.to(environment_probe.device)

        # WGAN div hyper parameters
        self.wk, self.wp = 2, 6

        # datasets
        # do not forget to rewrite
        folder = config.folder
        dataset = CHiME3(folder, 'train', config.n_sample,
                         config.sr)
        self.dataloader = DataLoader(
            dataset, config.batch_size, True, num_workers=config.num_workers, pin_memory=True, drop_last=True)
        logging.info(
            f'dataset | folder: {str(folder)} | size: {len(self.dataloader) * config.batch_size}')

    def train_discriminator(self, gt: Tensor, vi: Tensor) -> Tensor:
        logging.debug('train phase discriminator')
        # switch to train mode
        self.discriminator.train()

        # judge value towards real & fake
        real_v = torch.squeeze(self.discriminator(gt))
        fake_v = torch.squeeze(self.discriminator(vi))

        # loss calculate
        real_l, fake_l = -real_v.mean(), fake_v.mean()
        div = div_loss(self.discriminator, gt, vi,
                       self.environment_probe.device, self.wp)
        loss = real_l + fake_l + self.wk * div

        # backward
        self.opt_discriminator.zero_grad()
        loss.backward()
        self.opt_discriminator.step()

        self.discriminator.eval()

        return loss.item()

    # def train_dis_phase(self, gt: Tensor, vi: Tensor) -> Tensor:
    #     logging.debug('train phase discriminator')
    #     # switch to train mode
    #     self.dis_phase.train()

    #     # judge value towards real & fake
    #     real_v = torch.squeeze(self.dis_phase(gt))
    #     fake_v = torch.squeeze(self.dis_phase(vi))

    #     # loss calculate
    #     real_l, fake_l = -real_v.mean(), fake_v.mean()
    #     div = div_loss(self.dis_phase, gt, vi, self.wp)
    #     loss = real_l + fake_l + self.wk * div

    #     # backward
    #     self.opt_dis_phase.zero_grad()
    #     loss.backward()
    #     self.opt_dis_phase.step()

    #     return loss.item()

    # def train_dis_amplitude(self, gt: Tensor, vi: Tensor) -> Tensor:
    #     logging.debug('train amplitude discriminator')
    #     # switch to train mode
    #     self.dis_amplitude.train()

    #     # judge value towards real & fake
    #     real_v = torch.squeeze(self.dis_amplitude(gt))
    #     fake_v = torch.squeeze(self.dis_amplitude(vi))

    #     # loss calculate
    #     real_l, fake_l = -real_v.mean(), fake_v.mean()
    #     div = div_loss(self.dis_amplitude, gt, vi, self.wp)
    #     loss = real_l + fake_l + self.wk * div

    #     # backward
    #     self.opt_dis_amplitude.zero_grad()
    #     loss.backward()
    #     self.opt_dis_amplitude.step()

    #     return loss.item()

    def train_generator(self, c1: Tensor, c2: Tensor, gt: Tensor) -> dict:
        """
        Train generator 'c1 + c2 -> vir'
        """

        logging.debug('train generator')
        self.generator.train()

        vir = self.generator(c1, c2)

        # calculate loss towards criterion
        # b1 can be deleted, or it can weight the PESQ or something else
        b1, b2, b3 = self.config.weight  # b2 * l1 + b3 * adv

        l_l1 = b2 * self.l1(vir, gt)
        l_l1 = l_l1.mean()

        self.discriminator.eval()
        l_adv = -self.discriminator(vir).mean()  # judge phase: vir_phase

        loss = l_l1 + b3 * l_adv

        # backward
        self.opt_generator.zero_grad()
        loss.backward()
        self.opt_generator.step()

        # loss state
        state = {
            'g_loss': loss.item(),
            'g_l1': l_l1.item(),
            'g_adv': l_adv.item()
        }

        self.generator.eval()

        return state

    def model_load(self, mode: str):
        path = Path.cwd()/f'cache/{mode}.pth'
        logging.info('load the dict of ' + str(path))

        check_point = torch.load(path)
        self.generator.load_state_dict(check_point['g'])
        self.discriminator.load_state_dict(check_point['d'])

        self.opt_generator.load_state_dict(check_point['opt']['g'])
        self.opt_discriminator.load_state_dict(check_point['opt']['d'])

    def pretrain(self):
        logging.info("pretraining")
        for i in range(self.config.depth):
            process = tqdm(enumerate(self.dataloader),
                           disable=not self.config.debug)
            for idx, sample in process:
                sample = sample.to(self.environment_probe.device)
                c1, gt, c2 = torch.chunk(sample, 3, dim=1)
                vir = self.generator.pretrain(c1, c2, i)
                loss = self.l1(vir, gt).mean()
                process.set_description(f'L1 loss: {loss.item():03f}')
                self.opt_generator.zero_grad()
                loss.backward()
                self.opt_generator.step()

    def train_generator_only(self):
        max = 1e-8

        self.pretrain()
        for i in range(self.config.epochs + 1):
            process = tqdm(enumerate(self.dataloader),
                           disable=not self.config.debug)
            meter = AverageMeter()
            for idx, sample in process:
                sample = sample.to(self.environment_probe.device)
                c1, gt, c2 = torch.chunk(sample, 3, dim=1)
                vir = self.generator(c1, c2)
                loss = self.l1(vir, gt).mean()
                process.set_description(f'L1 loss: {loss.item():03f}')
                self.opt_generator.zero_grad()
                loss.backward()
                self.opt_generator.step()
                meter.update(loss.item())
            print(f'L1 avg: {meter.avg:03f}')
            eval = Eval(self.generator, self.config, eval=True)
            sdr = eval()
            self.scheduler.step(sdr)
            if sdr > max:
                max = sdr
                self.save('best_')
            self.save('last_')

    def cotrain(self):
        max = 1e-8

        for epoch in range(1, self.config.epochs + 1):
            process = tqdm(enumerate(self.dataloader),
                           disable=not self.config.debug)
            meter = AverageMeter()
            for idx, sample in process:
                sample = sample.to(self.environment_probe.device)
                c1, gt, c2 = torch.chunk(sample, 3, dim=1)
                d_loss = 0
                for i in range(5):
                    vir = self.generator(c1, c2)
                    d_loss = self.train_discriminator(gt, vir)
                g_loss = self.train_generator(c1, c2, gt)

                process.set_description(
                    f'g: {g_loss["g_loss"]:03f}, g_l1: {g_loss["g_l1"]:03f}, g_adv: {g_loss["g_adv"]:03f} | d: {d_loss:03f}')
                meter.update(Tensor(list(g_loss.values()) + [d_loss]))

            keys = ['g_loss', 'g_l1', 'g_adv']

            state = reduce(lambda x, y: dict(x, **y), [
                           {k: v} for k, v in zip(keys, meter.avg)])
            print(state)
            eval = Eval(self.generator, self.config, eval=True)
            sdr = eval()
            self.scheduler.step(sdr)
            if sdr > max:
                max = sdr
                self.save('best_cotrain')
            self.save('last_cotrain')

    def run(self):
        self.cotrain()
        # self.train_generator_only()

    def save(self, mode: str):
        path = Path(self.config.cache)
        path = Path.cwd() / path
        path.mkdir(parents=True, exist_ok=True)
        cache = path / f'{mode}.pth'
        logging.info(f'save checkpoint to {str(cache)}')
        state = {
            'g': self.generator.state_dict(),
            'd': self.discriminator.state_dict(),
            'opt': {
                'g': self.opt_generator.state_dict(),
                'd': self.opt_discriminator.state_dict()
            },
        }
        torch.save(state, cache)
