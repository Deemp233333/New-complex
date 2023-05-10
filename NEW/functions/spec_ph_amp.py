import torch
from torch import Tensor, stft, unsqueeze, abs, angle, atan2


def spec_ph_amp(signal: Tensor, n_fft: int = 1024):
    spec = stft(signal, n_fft=n_fft, return_complex=True).unsqueeze(1)
    amp = abs(spec)
    ph = atan2(spec.imag, replace_denormals(spec.real))
    return spec, amp, ph


def replace_denormals(x: torch.tensor, threshold=1e-10):
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y
