import os
import json
import argparse
import random
from datetime import datetime
from collections import defaultdict
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset
from transformers import ASTModel, TrainingArguments, Trainer
from safetensors.torch import save_file

# ========== CONFIG ==========
PRETRAINED_MODEL    = "MIT/ast-finetuned-audioset-10-10-0.4593"
CHUNKS_DIR          = "preprocessed_features"
MODEL_OUTPUT_DIR    = "ast_triplet_output"
BATCH_SIZE          = 2
EPOCHS              = 30
LEARNING_RATE       = 1e-4
TEST_RUN_FRACTION   = 0.01  # fraction to use when --test_run is set

# ========== DATA LOADING & SPLITTING ==========

def get_triplet_filepaths(root_dir):
    triplets = []
    for subdir in sorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(sub_path) or subdir == "logs":
            continue
        files = sorted([f for f in os.listdir(sub_path) if f.endswith(".pt")])
        groups = [files[i:i+3] for i in range(0, len(files), 3)]
        for group in groups:
            if len(group) < 3:
                continue
            a, p, n = group
            triplets.append((
                os.path.join(sub_path, a),
                os.path.join(sub_path, p),
                os.path.join(sub_path, n),
                subdir
            ))
    return triplets


def sanitize_dict_tensor(d):
    """Squeeze out leading channel dim if present, convert lists to tensors."""
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            v = v[0]
        elif isinstance(v, list):
            v = torch.tensor(v, dtype=torch.float32)
        if isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] == 1:
            v = v.squeeze(0)
        cleaned[k] = v
    return cleaned


class TripletFeatureDataset(TorchDataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a_path, p_path, n_path, _ = self.triplets[idx]
        a_dict = torch.load(a_path, weights_only=False)
        p_dict = torch.load(p_path, weights_only=False)
        n_dict = torch.load(n_path, weights_only=False)
        return {
            "anchor_input":   sanitize_dict_tensor(a_dict),
            "positive_input": sanitize_dict_tensor(p_dict),
            "negative_input": sanitize_dict_tensor(n_dict),
            "labels": 0
        }


def collate_fn(batch):
    def stack_dict(key):
        out = {}
        inner_keys = batch[0][key].keys()
        for ik in inner_keys:
            tensors = []
            for item in batch:
                v = item[key][ik]
                if isinstance(v, list):
                    v = torch.tensor(v)
                if v.ndim == 3 and v.shape[0] == 1:
                    v = v.squeeze(0)
                tensors.append(v)
            out[ik] = torch.stack(tensors, dim=0)
        return out

    return {
        "anchor_input":   stack_dict("anchor_input"),
        "positive_input": stack_dict("positive_input"),
        "negative_input": stack_dict("negative_input"),
        "labels":         torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    }


class ASTTripletWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.ast = base_model
        self.projector = nn.Sequential(
            nn.Linear(self.ast.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def embed(self, inputs):
        outputs = self.ast(**inputs)
        pooled = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(self.projector(pooled), dim=1)

    def forward(self, anchor_input, positive_input, negative_input, labels=None):
        emb_a = self.embed(anchor_input)
        emb_p = self.embed(positive_input)
        emb_n = self.embed(negative_input)
        d_ap = 1 - F.cosine_similarity(emb_a, emb_p)
        d_an = 1 - F.cosine_similarity(emb_a, emb_n)
        loss = torch.clamp(d_ap - d_an + 0.3, min=0.0).mean()
        logits = torch.stack([d_ap, d_an], dim=1)
        return {"loss": loss, "logits": logits}


def compute_metrics(eval_pred):
    distances = eval_pred.predictions
    preds = np.argmin(distances, axis=1)
    labels = eval_pred.label_ids
    acc = float((preds == labels).mean()) if labels is not None else 0.0
    return {"accuracy": acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_run", action="store_true",
                        help="Run a quick smoke test on a small subset")
    args = parser.parse_args()
    test_run = args.test_run

    # 1. Gather all triplets
    all_triplets = get_triplet_filepaths(CHUNKS_DIR)

    # 2. Stratified split by subgenre
    by_sub = defaultdict(list)
    for tpl in all_triplets:
        by_sub[tpl[3]].append(tpl)

    train_triplets, test_triplets = [], []
    for sub, items in by_sub.items():
        random.shuffle(items)
        n_test = max(1, int(len(items) * 0.1))
        test_triplets.extend(items[:n_test])
        train_triplets.extend(items[n_test:])

    # 3. Optionally shrink for quick debug run
    if test_run:
        n_tr = max(1, int(len(train_triplets) * TEST_RUN_FRACTION))
        n_te = max(1, int(len(test_triplets)  * TEST_RUN_FRACTION))
        train_triplets = random.sample(train_triplets, n_tr)
        test_triplets  = random.sample(test_triplets,  n_te)
        print(f"Test run: {len(train_triplets)} train / {len(test_triplets)} eval")

    # 4. Persist splits
    split_dir = os.path.join(MODEL_OUTPUT_DIR, "splits")
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "train_split.json"), "w") as f:
        json.dump(train_triplets, f, indent=2)
    with open(os.path.join(split_dir, "test_split.json"), "w") as f:
        json.dump(test_triplets, f, indent=2)

    # 5. Build datasets
    train_ds = TripletFeatureDataset(train_triplets)
    test_ds  = TripletFeatureDataset(test_triplets)

    # 6. Load optional feature‐stats (for normalization elsewhere)
    stats_path = os.path.join(CHUNKS_DIR, "feature_stats.json")
    ds_stats = None
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            ds_stats = json.load(f)

    # 7. Initialize model & trainer
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = ASTModel.from_pretrained(PRETRAINED_MODEL)
    model      = ASTTripletWrapper(base_model).to(device)

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="no",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=6,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_strategy="no",
        report_to="none",
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        warmup_ratio=0.1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )

    # 8. Train & save
    print("Starting training…")
    trainer.train()

    # Save final model and related artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("fully_trained_models_" + timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Save model weights in safetensors format (AST + projection head)
    save_file(model.state_dict(), os.path.join(save_dir, "model.safetensors"))

    # Save hyperparameters
    hyperparams = {
        "pretrained_model": PRETRAINED_MODEL,
        "chunks_dir": CHUNKS_DIR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "test_run": test_run,
        **training_args.to_dict()
    }
    with open(os.path.join(save_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    # Save train/test split info
    split_dir = os.path.join(MODEL_OUTPUT_DIR, "splits")
    for split_file in ["train_split.json", "test_split.json"]:
        shutil.copy(os.path.join(split_dir, split_file), os.path.join(save_dir, split_file))
