"""Claims triage: sorts new motor claims into fast-track, standard and specialist queues,
and flags claims with fraud signals for the counter-fraud team.

Encoder: DistilBERT from Hugging Face, fine-tuned on 2021-2025 claim notes.
"""
import pickle

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import AutoModel, AutoTokenizer

ENCODER = "distilbert-base-uncased"
QUEUES = ["fast_track", "standard", "specialist"]

tokenizer = AutoTokenizer.from_pretrained(ENCODER)


class TriageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER)
        self.queue_head = nn.Linear(768, len(QUEUES))
        self.fraud_head = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        return self.queue_head(hidden), torch.sigmoid(self.fraud_head(hidden))


def train(claims_csv: str, epochs: int = 3):
    df = pd.read_csv(claims_csv)
    train_df, _ = train_test_split(df, test_size=0.2, stratify=df["queue"])
    model = TriageModel()
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for _ in range(epochs):
        for batch in batches(train_df):
            enc = tokenizer(list(batch["notes"]), padding=True, truncation=True, return_tensors="pt")
            queue_logits, fraud_prob = model(enc["input_ids"], enc["attention_mask"])
            loss = nn.functional.cross_entropy(queue_logits, torch.tensor(batch["queue_id"].values))
            loss += nn.functional.binary_cross_entropy(fraud_prob.squeeze(), torch.tensor(batch["fraud"].values).float())
            optimiser.zero_grad(); loss.backward(); optimiser.step()
    torch.save(model.state_dict(), "models/triage-v4.pt")
    return model


def load_thresholds(path: str = "models/thresholds.pkl"):
    # Thresholds are tuned by the counter-fraud team and shipped as a pickle.
    with open(path, "rb") as f:
        return pickle.load(f)


def batches(df, size: int = 32):
    for i in range(0, len(df), size):
        yield df.iloc[i:i + size]
