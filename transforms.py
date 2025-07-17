import torchaudio.functional as F
import torchaudio.transforms as T
import torch
import random

class RandomAudioAugmentation:
    def __init__(self, sample_rate=16000, pitch_range=(-2, 2), noise_prob=0.3):
        self.sample_rate = sample_rate
        self.pitch_range = pitch_range
        self.noise_prob = noise_prob

    def __call__(self, waveform):
        if random.random() < 0.5:
            n_steps = random.uniform(*self.pitch_range)
            waveform = F.pitch_shift(waveform, self.sample_rate, n_steps)

        if random.random() < self.noise_prob:
            noise = torch.randn_like(waveform) * 0.01  # small noise
            waveform = waveform + noise

        return waveform