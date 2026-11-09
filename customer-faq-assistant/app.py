"""Customer FAQ assistant: answers policyholder questions from published policy documents.

Requests go through the VerifyWise AI Gateway, so every call is logged and billed
against the "Customer FAQ app" virtual key.
"""
import os

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

from rag import retrieve_context

GATEWAY_URL = os.environ["VERIFYWISE_GATEWAY_URL"]   # e.g. https://<host>/v1
VIRTUAL_KEY = os.environ["VERIFYWISE_VIRTUAL_KEY"]   # sk-vw-..., from the secrets store

client = OpenAI(base_url=GATEWAY_URL, api_key=VIRTUAL_KEY)
app = FastAPI(title="Northbridge customer FAQ assistant")

SYSTEM_PROMPT = (
    "You answer questions from Northbridge Insurance customers. Use only the policy extracts "
    "provided. If the answer is not in them, say so and offer to connect the customer to an agent. "
    "Never give advice on a claim decision."
)

HANDOFF_TOPICS = ("complaint", "dispute", "bereavement", "vulnerable")


class Question(BaseModel):
    text: str
    policy_type: str = "home"


@app.post("/ask")
def ask(q: Question):
    if any(word in q.text.lower() for word in HANDOFF_TOPICS):
        return {"answer": None, "handoff": True}

    context = retrieve_context(q.text, policy_type=q.policy_type)
    response = client.chat.completions.create(
        model="customer-faq-assistant",          # the gateway endpoint slug
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Policy extracts:\n{context}\n\nQuestion: {q.text}"},
        ],
        max_tokens=400,
        timeout=20,
    )
    return {"answer": response.choices[0].message.content, "handoff": False}
