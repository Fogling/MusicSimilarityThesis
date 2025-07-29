import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import ASTFeatureExtractor, ASTModel, TrainingArguments, Trainer
from transformers.feature_extraction_utils import BatchFeature
import torch.serialization
torch.serialization.add_safe_globals({"BatchFeature": BatchFeature})

# ========== CONFIG ==========
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
CHUNKS_DIR = "preprocessed_features"
MODEL_OUTPUT_DIR = "ast_triplet_output"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
EPOCHS = 10
MARGIN = 0.3

# ========== DATA LOADING AND SANITIZATION ==========
def unwrap_and_squeeze(d):
    cleaned = {}
    for k, v in d.items():
        if isinstance(v, list) and isinstance(v[0], torch.Tensor):
            v = v[0]
        elif isinstance(v, list):
            v = torch.tensor(v)

        if isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] == 1:
            v = v.squeeze(0)

        cleaned[k] = v
    return cleaned

def load_triplet_dataset(root_dir):
    anchors, positives, negatives = [], [], []
    for subdir in sorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(sub_path):
            continue
        files = sorted([f for f in os.listdir(sub_path) if f.endswith(".pt")])
        triplets = [files[i:i+3] for i in range(0, len(files), 3)]
        for triplet in triplets:
            if len(triplet) < 3:
                continue
            a, p, n = triplet
            a_dict = torch.load(os.path.join(sub_path, a), weights_only=False)
            p_dict = torch.load(os.path.join(sub_path, p), weights_only=False)
            n_dict = torch.load(os.path.join(sub_path, n), weights_only=False)

            anchors.append(unwrap_and_squeeze(a_dict))
            positives.append(unwrap_and_squeeze(p_dict))
            negatives.append(unwrap_and_squeeze(n_dict))

    return Dataset.from_dict({
        "anchor_input": anchors,
        "positive_input": positives,
        "negative_input": negatives
    })

# ========== MODEL WITH TRIPLET LOSS ==========
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

    def forward(self, anchor_input, positive_input, negative_input):
        emb_a = self.embed(anchor_input)
        emb_p = self.embed(positive_input)
        emb_n = self.embed(negative_input)

        d_ap = 1 - F.cosine_similarity(emb_a, emb_p)
        d_an = 1 - F.cosine_similarity(emb_a, emb_n)
        loss = torch.clamp(d_ap - d_an + MARGIN, min=0.0).mean()
        return {"loss": loss}

# ========== DATA COLLATOR ==========
def collate_triplets(batch):
    def stack_inputs(key):
        keys = batch[0][key].keys()
        stacked = {}
        for k in keys:
            tensors = []
            for item in batch:
                value = item[key][k]
                # If it's a list: convert to tensor
                if isinstance(value, list):
                    value = torch.tensor(value)
                # If it's still not tensor: raise clear error
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"Expected tensor but got {type(value)} for key {key}.{k}")
                # Squeeze unnecessary dimensions (like [1, 1024, 128])
                if value.ndim == 3 and value.shape[0] == 1:
                    value = value.squeeze(0)
                tensors.append(value)
            stacked[k] = torch.stack(tensors)
        return stacked

    return {
        "anchor_input": stack_inputs("anchor_input"),
        "positive_input": stack_inputs("positive_input"),
        "negative_input": stack_inputs("negative_input")
    }

# ========== MAIN ==========
def main():
    print("Loading triplet dataset...")
    dataset = load_triplet_dataset(CHUNKS_DIR)
    dataset = dataset.train_test_split(test_size=0.1)

    print("Loading AST model and feature extractor...")
    feature_extractor = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL)
    model = ASTModel.from_pretrained(PRETRAINED_MODEL)
    wrapped_model = ASTTripletWrapper(model).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        eval_strategy="no",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=os.path.join(MODEL_OUTPUT_DIR, "logs"),
        dataloader_num_workers=3,
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=wrapped_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collate_triplets
    )

    print("Starting triplet training...")
    trainer.train()
    trainer.save_model()
    print("Triplet model saved.")

if __name__ == "__main__":
    main()