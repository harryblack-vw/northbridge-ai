# Northbridge Insurance: AI services

> **Demo repository.** Northbridge Insurance is a fictional company. This code exists to show what VerifyWise AI Detection finds when it scans a repository. It is not production code, and the credential in `underwriting-copilot/tests/` is a fake placeholder.

Three services, one folder each:

| Folder | What it does | Owner |
|---|---|---|
| `customer-faq-assistant/` | Website and app chat that answers policyholder questions from published policy documents. Calls Gemini through the VerifyWise AI Gateway, retrieves from a Qdrant vector store. | Customer Service |
| `claims-triage/` | Sorts incoming motor claims by complexity and fraud signals. PyTorch classifier on a DistilBERT encoder. | Data Science |
| `underwriting-copilot/` | Agent that drafts underwriting notes and looks up policy records through MCP tools. Pilot. | Underwriting |

The nightly evaluation workflow is in `.github/workflows/nightly-eval.yml`.
