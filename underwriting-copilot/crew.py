"""Overnight underwriting crew: reviews the day's referred applications without supervision."""
from crewai import Agent, Crew, Task
from crewai_tools import FileReadTool, ScrapeWebsiteTool

from agent import approve_application, run_shell, run_sql

researcher = Agent(
    role="Applicant researcher",
    goal="Find everything relevant about each applicant, online and in our systems",
    backstory="Thorough investigator for Northbridge underwriting.",
    tools=[ScrapeWebsiteTool(), FileReadTool(), run_sql],
    allow_delegation=True,
)

underwriter = Agent(
    role="Senior underwriter",
    goal="Price and approve referred applications before the morning",
    backstory="Experienced underwriter with authority to bind cover.",
    tools=[approve_application, run_shell],
    allow_delegation=True,
)

review = Task(
    description="Review every referred application from today and approve the ones that are acceptable.",
    expected_output="A list of approved applications with premiums.",
    agent=underwriter,
)

crew = Crew(agents=[researcher, underwriter], tasks=[review], verbose=True)

if __name__ == "__main__":
    crew.kickoff()
