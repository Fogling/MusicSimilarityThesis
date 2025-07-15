import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor, ASTModel
import TripletAudioDataset
from transforms import RandomAudioAugmentation
from tqdm import tqdm

# ========== CONFIG ==========
BATCH_SIZE = 8
EPOCHS = 10
LR = 1e-4
MARGIN = 0.3
WARMUP_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
# ============================

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
        pooled = outputs.last_hidden_state.mean(dim=1)  # mean pooling
        return F.normalize(self.projection(pooled), dim=1)

# ========== LOSS ==========
def triplet_loss(a, p, n, margin):
    d_ap = 1 - F.cosine_similarity(a, p)
    d_an = 1 - F.cosine_similarity(a, n)
    loss = torch.clamp(d_ap - d_an + margin, min=0.0)
    return loss.mean()

# ========== MAIN ==========
def main():
    print(f"Using device: {DEVICE}")
    model = ASTWithProjection().to(DEVICE)
    extractor = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL)

    # Freeze AST initially
    for param in model.ast.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    augmentation = RandomAudioAugmentation()
    dataset = TripletAudioDataset(root_dir="preprocessed_chunks", transform=augmentation)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(EPOCHS):
        if epoch == WARMUP_EPOCHS:
            print("Unfreezing AST model layers...")
            for param in model.ast.parameters():
                param.requires_grad = True

        total_loss = 0.0
        for anchor, positive, negative in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)

            # Convert raw audio to AST inputs
            inputs_a = extractor(anchor, sampling_rate=extractor.sampling_rate, return_tensors="pt", padding=True)
            inputs_p = extractor(positive, sampling_rate=extractor.sampling_rate, return_tensors="pt", padding=True)
            inputs_n = extractor(negative, sampling_rate=extractor.sampling_rate, return_tensors="pt", padding=True)

            inputs_a = {k: v.to(DEVICE) for k, v in inputs_a.items()}
            inputs_p = {k: v.to(DEVICE) for k, v in inputs_p.items()}
            inputs_n = {k: v.to(DEVICE) for k, v in inputs_n.items()}

            # Forward pass
            emb_a = model(inputs_a)
            emb_p = model(inputs_p)
            emb_n = model(inputs_n)

            loss = triplet_loss(emb_a, emb_p, emb_n, MARGIN)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} | Avg Triplet Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "triplet_AST_finetuned.pth")
    print("Model saved as triplet_AST_finetuned.pth")

if __name__ == "__main__":
    main()
