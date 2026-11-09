# Northbridge Insurance: AI services

> **Demo repository.** Northbridge Insurance is a fictional company. This code exists to show what VerifyWise AI Detection finds when it scans a repository. It is not production code. Every credential in it, including `config/settings.py` and `underwriting-copilot/tests/`, is a fake placeholder that does not work.

Three services, one folder each:

| Folder | What it does | Owner |
|---|---|---|
| `customer-faq-assistant/` | Website and app chat that answers policyholder questions from published policy documents. Calls Gemini through the VerifyWise AI Gateway, retrieves from a Qdrant vector store. | Customer Service |
| `claims-triage/` | Sorts incoming motor claims by complexity and fraud signals. PyTorch classifier on a DistilBERT encoder. | Data Science |
| `underwriting-copilot/` | Agent that drafts underwriting notes and looks up policy records through MCP tools. Pilot. | Underwriting |

The nightly evaluation workflow is in `.github/workflows/nightly-eval.yml`.
