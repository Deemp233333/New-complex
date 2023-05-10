from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import os
import torch
import json
import torchaudio
from torch import nn
import pandas as pd


def select_low_correlation_signal(annotations):
    low_correlation_signal = []
    for i in range(len(annotations)):
        if annotations.iloc[i, 4] < 0.9 or annotations.iloc[i, 5] < 0.9 or annotations.iloc[i, 6] < 0.9:
            # utter_name = str(annotations.iloc[i, 0]).split('_',2)
            # low_correlation_signal.append(utter_name[1])
            low_correlation_signal.append(annotations.iloc[i, 0])

    return low_correlation_signal


class CHiME3(Dataset):
    def __init__(self, data_dir: str, mode: str, num_samples: int = 64000, sr: int = 16000, env: str = 'all'):
        super(CHiME3, self).__init__()
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.sr = sr
        self.mode = mode
        annotations = pd.read_csv(
            '/home/jiachen/Downloads/CHiME4_diff/CHiME3/data/annotations/mic_error.csv')
        self.low_correlation_signal = select_low_correlation_signal(
            annotations)

        # switch mode
        if mode == 'train':
            json_dir_real = 'tr05_real.json'
            json_dir_simu = 'tr05_simu.json'
            json_dir_bth = 'tr05_bth.json'
            file_dir = 'tr05'
        elif mode == 'test':
            json_dir_real = 'et05_real.json'
            json_dir_simu = 'et05_simu.json'
            json_dir_bth = 'et05_bth.json'
            file_dir = 'et05'
        elif mode == 'valid':
            json_dir_real = 'dt05_real.json'
            json_dir_simu = 'dt05_simu.json'
            json_dir_bth = 'dt05_bth.json'
            file_dir = 'dt05'
        else:
            raise ValueError('mode error')
        if env == 'all':
            jsons = [json_dir_real, json_dir_simu]
        elif env == 'simu':
            jsons = [json_dir_simu]
        else:
            jsons = [json_dir_real]

        self.path = {'json': [
            Path(self.data_dir) / 'annotations' / js for js in jsons], 'audio': []}
        speech_path = self.data_dir + '/audio/16kHz/isolated'
        # load jsons
        for js in self.path['json']:
            with js.open() as F:
                chime3_json = json.load(F)
            if 'real' in js.name:
                data_type = '_real'
            else:
                data_type = '_simu'
            # load speech path
            for file in chime3_json:
                wav_name = file['speaker'] + '_' + \
                    file['wsj_name'] + '_' + file['environment']
                if wav_name not in self.low_correlation_signal:
                    self.path['audio'].append(
                        speech_path + '/' + (file_dir + '_' + file['environment'].lower() + data_type) + '/' + wav_name)

    def __getitem__(self, index):
        speech_mul_ch = []
        for i in range(4, 7):
            speech, sr = torchaudio.load(
                Path(self.path['audio'][index] + '.CH' + str(i) + '.wav'))
            speech = self._resample_if_necessary(speech, sr)
            speech = self._cut_if_necessary(speech)
            speech = self._right_pad_if_necessary(speech)
            speech_mul_ch.append(speech)
        ret = torch.cat(speech_mul_ch, dim=0)
        ret = ret - torch.mean(ret)
        ret = ret/torch.max(torch.abs(speech))
        return ret

    def __len__(self):
        return len(self.path['audio'])

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            signal = resampler(signal)
        return signal

    def _extract_audio(self, speech, sr, start, end):
        speech = speech[start * sr:end * sr]
        return speech

# data_dir ='data/audio/16kHz/isolated/'
# dataset = CHiME3(data_dir, 'train', num_samples=64000, sr=16000)
# print(len(dataset))
# print(dataset[0])
# pdb.set_trace()
