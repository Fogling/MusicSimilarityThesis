from datasets.TripletAudioDataset import TripletAudioDataset
from torch.utils.data import DataLoader
from transforms import RandomAudioAugmentation

#augmentation = RandomAudioAugmentation(sample_rate=16000)
#dataset = TripletAudioDataset(root_dir="MP3", transform=augmentation)

dataset = TripletAudioDataset(root_dir="MP3", duration=10)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

anchor, positive, negative = next(iter(loader))
print(anchor.shape, positive.shape, negative.shape)
