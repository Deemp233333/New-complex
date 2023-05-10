import argparse
import logging
from argparse import Namespace
from pathlib import Path

import torch


from pipeline.train_complex import Train
from pipeline.eval import Eval
from pipeline.test import Test
from utils.environment_probe import EnvironmentProbe


def parse_args() -> Namespace:
    # args parser
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument(
        '--folder', default='/mnt/nas/jiachen/Downloads/ChiME3/data', help='data root path')
    parser.add_argument('--n_sample', default=64000,
                        help='thr number of the sampling')
    parser.add_argument('--sr', default=16000, help='thr sr of the sampling')

    parser.add_argument('--cache', default='cache',
                        help='weights cache folder')

    # NEW opt
    parser.add_argument('--g_depth', default=9, type=int,
                        help='generator network depth')
    parser.add_argument('--d_depth', default=5, type=int,
                        help='discriminator network depth')

    parser.add_argument('--weight', nargs='+', type=float,
                        default=[0.01, 1, 0.01], help='loss weight')
    parser.add_argument('--n_fft', default=1024, type=int, help='stft n-fft')

    # checkpoint opt
    parser.add_argument('--epochs', type=int, default=200,
                        help='epoch to train')
    # optimizer opt
    parser.add_argument('--optimizer', default='Adam',
                        help='choose the optimizer')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-3, help='learning rate')
    # dataloader opt
    parser.add_argument('--batch_size', type=int, default=16,
                        help='dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='dataloader workers number')

    # experimental opt
    parser.add_argument('--debug', default=True,
                        action='store_true', help='debug mode (default: off)')

    return parser.parse_args()


if __name__ == '__main__':

    config = parse_args()

    logging.basicConfig(level='INFO')

    environment_probe = EnvironmentProbe()
    train_process = Train(environment_probe, config)
    train_process.model_load('best_complex_cotrain')
    train_process.run()
    # test_process = Test(config=config, path=Path(
    #     '/home/qiuzheng/Desktop/project/NEW/cache/-01.pth'), eval=True, env='real')
    # test_process()
