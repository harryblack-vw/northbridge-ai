"""Policy database access for the copilot, plus the policy-db MCP client."""
import asyncio
import json
import sqlite3

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

db = sqlite3.connect("/srv/underwriting/policies.db", check_same_thread=False)
SERVER = StdioServerParameters(command="npx", args=["-y", "@northbridge/policy-db-mcp"])


async def _call(tool: str, **kwargs):
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, kwargs)
            return json.loads(result.content[0].text)


def fetch_applicant(application_id: str):
    return asyncio.run(_call("get_applicant", application_id=application_id))
