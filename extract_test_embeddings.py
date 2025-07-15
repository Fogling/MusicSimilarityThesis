import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ASTFeatureExtractor, ASTModel
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
TRAIN_DIR = "preprocessed_chunks_train"
TEST_DIR = "preprocessed_chunks_test"
MODEL_PATH = "triplet_AST_finetuned.pth"
OUTPUT_PATH = "test_similarity_scores.txt"
SUBGENRES = ["Chill House", "Banger House", "Emotional Techno", "Dark Techno", "Zyzz"]

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

def load_embeddings_from_dir(model, extractor, directory):
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for sub in SUBGENRES:
            genre_dir = os.path.join(directory, sub)
            if not os.path.isdir(genre_dir):
                continue
            embeddings[sub] = []
            for fname in os.listdir(genre_dir):
                if not fname.endswith(".pt"):
                    continue
                path = os.path.join(genre_dir, fname)
                waveform = torch.load(path).unsqueeze(0).to(DEVICE)
                inputs = extractor(waveform, sampling_rate=extractor.sampling_rate, return_tensors="pt", padding=True)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                emb = model(inputs)
                embeddings[sub].append((fname, emb.squeeze(0)))
    return embeddings

def compute_centroids(train_embeddings):
    centroids = {}
    for sub, embs in train_embeddings.items():
        stack = torch.stack([e for _, e in embs])
        centroids[sub] = F.normalize(stack.mean(dim=0), dim=0)
    return centroids

def main():
    print(f"Loading model from {MODEL_PATH}...")
    model = ASTWithProjection().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    extractor = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL)

    print("Loading train embeddings for centroid computation...")
    train_embeddings = load_embeddings_from_dir(model, extractor, TRAIN_DIR)
    centroids = compute_centroids(train_embeddings)

    print("Evaluating test set similarity...")
    test_embeddings = load_embeddings_from_dir(model, extractor, TEST_DIR)

    with open(OUTPUT_PATH, "w") as out:
        for sub, entries in test_embeddings.items():
            for fname, emb in entries:
                out.write(f"{fname} (true: {sub}):\n")
                sims = {}
                for target_sub, centroid in centroids.items():
                    sim = F.cosine_similarity(emb, centroid, dim=0).item()
                    sims[target_sub] = sim
                # Normalize to pseudo-percentages
                total = sum([v for v in sims.values()])
                for k in sims:
                    sims[k] = round(100 * sims[k] / total, 2) if total > 0 else 0.0
                for target_sub in SUBGENRES:
                    out.write(f"  {target_sub}: {sims.get(target_sub, 0.0):.2f}%\n")
                out.write("\n")

    print(f"Done! Output written to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
