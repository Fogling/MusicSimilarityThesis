import os
import random
import torch
from torch.utils.data import Dataset

class TripletAudioDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to directory containing subgenre folders with preprocessed AST inputs.
        """
        self.root_dir = root_dir
        self.subgenre_to_files = self._gather_files()
        self.all_files = [(g, f) for g, fs in self.subgenre_to_files.items() for f in fs]
        self.subgenres = list(self.subgenre_to_files.keys())

    def _gather_files(self):
        mapping = {}
        for sub in os.listdir(self.root_dir):
            path = os.path.join(self.root_dir, sub)
            if not os.path.isdir(path):
                continue
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]
            if len(files) >= 2:
                mapping[sub] = files
        return mapping

    def _load_features(self, path):
        features = torch.load(path)
        return dict(features)  # <-- this line fixes it

    def __getitem__(self, index):
        anchor_genre, anchor_path = self.all_files[index]

        pos_list = [f for f in self.subgenre_to_files[anchor_genre] if f != anchor_path]
        positive_path = random.choice(pos_list)

        other_genres = [g for g in self.subgenres if g != anchor_genre]
        negative_genre = random.choice(other_genres)
        negative_path = random.choice(self.subgenre_to_files[negative_genre])

        sample = self._load_features(anchor_path)
        print("Sample type:", type(sample))  # should be <class 'dict'>
        print("Sample keys:", sample.keys()) # should include 'input_values', etc.

        return (
            self._load_features(anchor_path),
            self._load_features(positive_path),
            self._load_features(negative_path),
        )

    def __len__(self):
        return len(self.all_files)
