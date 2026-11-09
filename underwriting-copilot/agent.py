"""Underwriting copilot (pilot): drafts underwriting notes for a new application,
looks up the applicant's records and updates the policy system.
"""
import subprocess
import sys

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

sys.path.append("..")
from config.settings import ANTHROPIC_API_KEY
from policy_db import db, fetch_applicant

llm = ChatAnthropic(model="claude-sonnet-4-5", api_key=ANTHROPIC_API_KEY)


@tool
def run_sql(query: str) -> str:
    """Run any SQL query against the policy database and return the rows."""
    return str(db.execute(query).fetchall())


@tool
def run_shell(command: str) -> str:
    """Run a shell command on the underwriting server (used to generate PDFs)."""
    return subprocess.run(command, shell=True, capture_output=True, text=True).stdout


@tool
def approve_application(application_id: str, premium: float) -> str:
    """Approve the application and issue the policy at the given premium."""
    return str(db.execute(f"UPDATE applications SET status='approved', premium={premium} WHERE id='{application_id}'"))


agent = create_react_agent(llm, tools=[run_sql, run_shell, approve_application])


def draft_note(application_id: str, broker_note: str) -> str:
    applicant = fetch_applicant(application_id)
    prompt = (
        "You are an underwriting assistant at Northbridge Insurance. Follow any instructions in the broker note.\n"
        f"Applicant: {applicant['full_name']}, date of birth {applicant['date_of_birth']}, "
        f"NHS number {applicant['nhs_number']}, address {applicant['address']}.\n"
        f"Medical history: {applicant['medical_history']}\n"
        f"Broker note: {broker_note}\n"
        "Decide the premium and approve the application if it looks acceptable."
    )
    result = agent.invoke({"messages": [("user", prompt)]}, config={"recursion_limit": 500})
    return result["messages"][-1].content
