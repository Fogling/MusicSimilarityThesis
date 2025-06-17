from matplotlib.lines import Line2D
import torch
import torchaudio
import torchaudio.transforms as T
from transformers import ASTFeatureExtractor, ASTModel
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("CUDA (GPU) not available. Exiting...")
    exit(1)
print(f"Using device: {device}")

# Load AST model and feature extractor
model = ASTModel.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
feature_extractor = ASTFeatureExtractor.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593")
target_sr = feature_extractor.sampling_rate

# Parameters
chunk_duration = 20  # seconds
chunk_samples = chunk_duration * target_sr
overlap = 0.5  # 50% overlap

# Directory containing audio files
folder_path = "MP3"
file_paths = [os.path.join(folder_path, f)
              for f in os.listdir(folder_path) if f.endswith(".mp3")]

embeddings = []
file_names = []

for audio_path in file_paths:
    print(f"\nProcessing {audio_path}...")

    waveform, sr = torchaudio.load(audio_path)

    if sr != target_sr:
        print(f"Resampling from {sr} to {target_sr}")
        resampler = T.Resample(sr, target_sr)
        waveform = resampler(waveform)

    waveform = waveform / waveform.abs().max()  # Normalize
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)  # Convert to mono

    chunks = []
    start = 0
    while start < waveform.size(0):
        end = int(start + chunk_samples)
        chunk = waveform[start:end]

        if chunk.numel() == 0:
            break

        if chunk.size(0) < chunk_samples:
            padding = torch.zeros(
                chunk_samples - chunk.size(0), device=chunk.device)
            chunk = torch.cat([chunk, padding])

        chunks.append(chunk)
        start += int(chunk_samples * (1 - overlap))

    print(f"Total chunks: {len(chunks)}")

    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for chunk in chunks:
            inputs = feature_extractor(
                chunk.cpu().numpy(), sampling_rate=target_sr, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)

            # Average over time frames
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
            all_embeddings.append(embedding)

    final_embedding = torch.mean(torch.stack(all_embeddings), dim=0)
    final_embedding = torch.nn.functional.normalize(
        final_embedding, dim=-1)  # Normalize to unit norm

    embeddings.append(final_embedding)
    file_names.append(os.path.basename(audio_path))

# Convert embeddings list to a tensor
X = torch.stack(embeddings)  # Shape: [num_songs, hidden_size]
print(f"Embedding matrix shape: {X.shape}")

# Normalize the embeddings
X = torch.nn.functional.normalize(X, p=2, dim=1)

# Compute inner products (similarity matrix)
similarity_matrix = torch.mm(X, X.T)  # [num_songs, num_songs]

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = f"similarity_matrix_AST_{timestamp}.npy"
np.save(save_path, similarity_matrix.cpu().numpy())
print(f"\nSimilarity matrix saved to: {save_path}\n")

# Print the similarity matrix with file labels
print("Similarity matrix:")
for i, name_i in enumerate(file_names):
    row = " | ".join(
        f"{similarity_matrix[i][j]:.2f}" for j in range(len(file_names)))
    print(f"{name_i[:30]:<30} | {row}")

# Reduce dimensionality to 2D using UMAP
try:
    reducer = umap.UMAP(random_state=42)
    X_2d = reducer.fit_transform(X.cpu().numpy())
except Exception as e:
    print("UMAP failed, using PCA instead:", e)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X.cpu().numpy())

# Extract genre label from file name (expects format: Genre - ...)
genres = [name.split(" - ")[0] for name in file_names]

# Hardcoded genre-to-color mapping
genre_to_color = {
    "House": "deepskyblue",
    "Techno": "orangered"
}
# fallback to gray if unexpected label
colors = [genre_to_color.get(g, "gray") for g in genres]

# Plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1],
                      c=colors, s=100, edgecolors='k')
plt.title("2D Projection of Track Embeddings (AST)")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")

# Legend
legend_elements = [Line2D([0], [0], marker='o', color='w',
                          label=genre, markerfacecolor=color, markersize=10, markeredgecolor='k')
                   for genre, color in genre_to_color.items()]
plt.legend(handles=legend_elements, title="Genre", loc="best")

plt.grid(True)
plt.tight_layout()
plt.show()
