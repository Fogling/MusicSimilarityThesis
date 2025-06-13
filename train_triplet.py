from datasets.triplet_audio_dataset import TripletAudioDataset
from torch.utils.data import DataLoader

dataset = TripletAudioDataset(root_dir="MP3", duration=10)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

anchor, positive, negative = next(iter(loader))
print(anchor.shape, positive.shape, negative.shape)
