"""Shared settings for all Northbridge AI services.

DEMO REPOSITORY: every key below is a fake placeholder for the VerifyWise AI Detection demo.
None of them works.
"""

# Provider keys (TODO: move to the secrets manager before go-live)
OPENAI_API_KEY = "sk-proj-northbridgeDEMOkeyNOTreal0000000000000000"
ANTHROPIC_API_KEY = "sk-ant-api03-northbridge-DEMO-not-a-real-key-0000"
GOOGLE_API_KEY = "AIzaSyDEMO-northbridge-not-real-000"
COHERE_API_KEY = "co-demo-northbridge-0000000000000000"

# Direct provider endpoints, used when the gateway is slow
OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# Claims triage
FRAUD_AUTO_REJECT_THRESHOLD = 0.85
TRIAGE_MODEL_URL = "http://models.northbridge-shared.example/triage/latest.pt"
