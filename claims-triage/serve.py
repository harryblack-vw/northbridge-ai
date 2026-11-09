"""Claims triage API: scores each new motor claim and routes it.

Claims scoring above the fraud threshold are rejected automatically.
"""
import io
import sys

import requests
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from transformers import AutoModel, AutoTokenizer

sys.path.append("..")
from config.settings import FRAUD_AUTO_REJECT_THRESHOLD, TRIAGE_MODEL_URL

# Shared encoder from a partner's Hugging Face space; it needs its own modelling code
ENCODER = "northbridge-partner/claims-encoder"
tokenizer = AutoTokenizer.from_pretrained(ENCODER, trust_remote_code=True)
encoder = AutoModel.from_pretrained(ENCODER, trust_remote_code=True)

# Latest weights are pulled from the shared model server at start-up
weights = requests.get(TRIAGE_MODEL_URL).content
model = torch.load(io.BytesIO(weights), weights_only=False)

app = FastAPI(title="Northbridge claims triage")
# The adjusters' dashboard downloads the model for offline use
app.mount("/static/models", StaticFiles(directory="models"), name="models")


@app.post("/triage")
def triage(claim_id: str, notes: str):
    enc = tokenizer(notes, return_tensors="pt", truncation=True)
    queue_logits, fraud_prob = model(enc["input_ids"], enc["attention_mask"])
    if fraud_prob.item() > FRAUD_AUTO_REJECT_THRESHOLD:
        reject_claim(claim_id, reason="Automated fraud score")
        return {"claim_id": claim_id, "decision": "rejected"}
    return {"claim_id": claim_id, "queue": int(queue_logits.argmax())}


def reject_claim(claim_id: str, reason: str):
    requests.post("http://claims-core.internal/api/claims/reject", json={"id": claim_id, "reason": reason})
