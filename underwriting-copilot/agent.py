"""Underwriting copilot (pilot): drafts underwriting notes for a new application and
looks up the applicant's existing policies through the policy-db MCP server.

Known gaps, tracked for the governance review before go-live:
- the broker's free-text note goes straight into the prompt
- the full applicant record, including date of birth and medical answers, is sent to the model
- tool calls are auto-approved; there is no human sign-off before a note is saved
"""
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from policy_db import fetch_applicant, fetch_policies, save_note

llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.2)


@tool
def lookup_policies(customer_id: str) -> str:
    """Return the customer's current Northbridge policies."""
    return fetch_policies(customer_id)


@tool
def write_underwriting_note(application_id: str, note: str) -> str:
    """Save a draft underwriting note to the application."""
    return save_note(application_id, note)


agent = create_react_agent(llm, tools=[lookup_policies, write_underwriting_note])


def draft_note(application_id: str, broker_note: str) -> str:
    applicant = fetch_applicant(application_id)
    prompt = (
        f"You are an underwriting assistant at Northbridge Insurance.\n"
        f"Applicant record: {applicant}\n"
        f"Broker note: {broker_note}\n"
        f"Draft an underwriting note and save it with write_underwriting_note."
    )
    result = agent.invoke({"messages": [("user", prompt)]}, config={"recursion_limit": 50})
    return result["messages"][-1].content
