import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor, ASTModel
from datasets.TripletAudioDataset import TripletAudioDataset
from transforms import RandomAudioAugmentation
from tqdm import tqdm
import random
import shutil

# ========== CONFIG ==========
BATCH_SIZE = 8
EPOCHS = 30
LOG_EVERY = 2
LR = 1e-4
MARGIN = 0.3
WARMUP_EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SUBGENRES = ["Chill House", "Banger House", "Emotional Techno", "Dark Techno", "Zyzz Music"]
CHUNKS_DIR = "preprocessed_chunks"
TRAIN_DIR = "preprocessed_chunks_train"
TEST_DIR = "preprocessed_chunks_test"
TEST_LOG = "test_set_log.txt"
CHUNKS_PER_TRACK = 3

# ========== MODEL ==========
class ASTWithProjection(nn.Module):
    def __init__(self, projection_dim=128):
        super().__init__()
        self.ast = ASTModel.from_pretrained(PRETRAINED_MODEL)
        self.projection = nn.Sequential(
            nn.Linear(self.ast.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, inputs):
        outputs = self.ast(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(self.projection(pooled), dim=1)

# ========== UTILS ==========
def prepare_balanced_data():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    with open(TEST_LOG, "w") as log:
        for sub in SUBGENRES:
            full_path = os.path.join(CHUNKS_DIR, sub)
            files = [f for f in os.listdir(full_path) if f.endswith("_chunk1.pt")]
            base_names = [f.replace("_chunk1.pt", "") for f in files]

            # Filter only tracks that have all chunks
            valid_tracks = []
            for name in base_names:
                all_exist = all(os.path.exists(os.path.join(full_path, f"{name}_chunk{i+1}.pt")) for i in range(CHUNKS_PER_TRACK))
                if all_exist:
                    valid_tracks.append(name)

            print(f"{sub}: {len(valid_tracks)} valid tracks with {CHUNKS_PER_TRACK} chunks")

            if len(valid_tracks) < 50:
                print(f"WARNING: {sub} only has {len(valid_tracks)} usable tracks (need 50)")

            selected = valid_tracks[:50]
            test_size = random.randint(5, 10)
            test_tracks = random.sample(selected, test_size)
            train_tracks = [t for t in selected if t not in test_tracks]

            os.makedirs(os.path.join(TRAIN_DIR, sub), exist_ok=True)
            os.makedirs(os.path.join(TEST_DIR, sub), exist_ok=True)

            for track in train_tracks:
                for i in range(CHUNKS_PER_TRACK):
                    fname = f"{track}_chunk{i+1}.pt"
                    shutil.copy(os.path.join(full_path, fname), os.path.join(TRAIN_DIR, sub, fname))

            for track in test_tracks:
                for i in range(CHUNKS_PER_TRACK):
                    fname = f"{track}_chunk{i+1}.pt"
                    shutil.copy(os.path.join(full_path, fname), os.path.join(TEST_DIR, sub, fname))

            log.write(f"{sub} test tracks (total {len(test_tracks)}):\n")
            for t in test_tracks:
                log.write(f"  - {t}\n")
            log.write("\n")

# ========== LOSS ==========
def triplet_loss(a, p, n, margin):
    d_ap = 1 - F.cosine_similarity(a, p)
    d_an = 1 - F.cosine_similarity(a, n)
    loss = torch.clamp(d_ap - d_an + margin, min=0.0)
    return loss.mean()

# ========== MAIN ==========
def main():
    print(f"Using device: {DEVICE}")

    # Optional: Skip re-prepping if folders already exist
    if not os.path.exists(TRAIN_DIR) or not os.path.exists(TEST_DIR):
        print("Preparing train/test data...")
        prepare_balanced_data()
    else:
        print("Skipping dataset prep — folders already exist.")

    model = ASTWithProjection().to(DEVICE)
    extractor = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL)

    for param in model.ast.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    augmentation = RandomAudioAugmentation()
    dataset = TripletAudioDataset(root_dir=TRAIN_DIR, transform=augmentation)
    print(f"Total training triplets available: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty. Check preprocessing and file structure.")

    model.train()
    for epoch in range(EPOCHS):
        if epoch == WARMUP_EPOCHS:
            print("Unfreezing AST model layers...")
            for param in model.ast.parameters():
                param.requires_grad = True

        total_loss = 0.0
        for anchor, positive, negative in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            
            inputs_a = extractor(
                [x.cpu().numpy().squeeze() for x in anchor],
                sampling_rate=extractor.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            inputs_p = extractor(
                [x.cpu().numpy().squeeze() for x in positive],
                sampling_rate=extractor.sampling_rate,
                return_tensors="pt",
                padding=True
            )
            inputs_n = extractor(
                [x.cpu().numpy().squeeze() for x in negative],
                sampling_rate=extractor.sampling_rate,
                return_tensors="pt",
                padding=True
            )

            inputs_a = {k: v.to(DEVICE) for k, v in inputs_a.items()}
            inputs_p = {k: v.to(DEVICE) for k, v in inputs_p.items()}
            inputs_n = {k: v.to(DEVICE) for k, v in inputs_n.items()}

            emb_a = model(inputs_a)
            emb_p = model(inputs_p)
            emb_n = model(inputs_n)

            loss = triplet_loss(emb_a, emb_p, emb_n, MARGIN)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % LOG_EVERY == 0:
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1} | Avg Triplet Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "triplet_AST_finetuned.pth")
    print("Model saved as triplet_AST_finetuned.pth")

if __name__ == "__main__":
    main()
