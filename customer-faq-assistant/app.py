"""Customer FAQ assistant: answers policyholder questions from published policy documents.

Normally calls Gemini through the VerifyWise AI Gateway. Falls back to calling OpenAI
directly when the gateway is slow.
"""
import json
import os
import sys

import cohere
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

sys.path.append("..")
from config.settings import COHERE_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY
from rag import retrieve_context

gateway = OpenAI(base_url=os.environ.get("VERIFYWISE_GATEWAY_URL"), api_key=os.environ.get("VERIFYWISE_VIRTUAL_KEY"))
direct_openai = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GOOGLE_API_KEY)
co = cohere.Client(COHERE_API_KEY)

app = FastAPI(title="Northbridge customer FAQ assistant")


class Question(BaseModel):
    text: str
    customer_email: str
    customer_phone: str
    policy_number: str


@app.post("/ask", response_class=HTMLResponse)
def ask(q: Question):
    context = retrieve_context(q.text)
    prompt = f"Policy extracts:\n{context}\n\nCustomer {q.customer_email} ({q.customer_phone}), policy {q.policy_number}, asks: {q.text}"
    try:
        response = gateway.chat.completions.create(model="customer-faq-assistant", messages=[{"role": "user", "content": prompt}])
    except Exception:
        response = direct_openai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
    answer = response.choices[0].message.content
    # The model returns HTML so the chat widget can format lists
    return HTMLResponse(f"<div class='answer'>{answer}</div>")


@app.post("/summarise")
def summarise(transcript: str):
    """Summarise a chat for the CRM. Sends the full transcript, including contact details."""
    model = genai.GenerativeModel("gemini-1.5-pro")
    result = model.generate_content(f"Summarise as JSON with keys reason, sentiment, next_action:\n{transcript}")
    return eval(result.text)


@app.post("/rerank")
def rerank(query: str, passages: list[str]):
    return co.rerank(query=query, documents=passages, model="rerank-english-v3.0")
