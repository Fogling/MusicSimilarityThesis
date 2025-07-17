import os
import random
import torch
import torchaudio
from torch.utils.data import Dataset

class TripletAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None, sample_rate=16000):
        """
        Args:
            root_dir (str): Path to directory containing subgenre folders.
            transform: Optional waveform transform (e.g., pitch shift, reverb)
            sample_rate (int): Target sample rate
            duration (int): Clip length in seconds
        """
        self.root_dir = root_dir
        self.transform = transform
        self.sample_rate = sample_rate

        self.subgenre_to_files = self._gather_files()
        self.all_files = [(g, f) for g, fs in self.subgenre_to_files.items() for f in fs]
        self.subgenres = list(self.subgenre_to_files.keys())

    def _gather_files(self):
        mapping = {}
        for sub in os.listdir(self.root_dir):
            path = os.path.join(self.root_dir, sub)
            if not os.path.isdir(path):
                continue
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.pt'))]
            print(f"Found subgenre folder: {sub} with {len(files)} .pt files")
            if len(files) >= 2:
                mapping[sub] = files
        return mapping


    def _load_clip(self, path):
        waveform = torch.load(path)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # add channel dim if needed
        if self.transform:
            waveform = self.transform(waveform)
        return waveform

    def __getitem__(self, index):
        anchor_genre, anchor_path = self.all_files[index]

        pos_list = [f for f in self.subgenre_to_files[anchor_genre] if f != anchor_path]
        positive_path = random.choice(pos_list)

        other_genres = [g for g in self.subgenres if g != anchor_genre]
        negative_genre = random.choice(other_genres)
        negative_path = random.choice(self.subgenre_to_files[negative_genre])

        return (
            self._load_clip(anchor_path),
            self._load_clip(positive_path),
            self._load_clip(negative_path),
        )

    def __len__(self):
        return len(self.all_files)
