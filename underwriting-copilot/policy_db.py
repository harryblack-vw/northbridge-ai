"""Thin client for the policy-db MCP server (see mcp.json)."""
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
import asyncio
import json

SERVER = StdioServerParameters(command="npx", args=["-y", "@northbridge/policy-db-mcp"])


async def _call(tool: str, **kwargs):
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool, kwargs)
            return json.loads(result.content[0].text)


def fetch_applicant(application_id: str):
    return asyncio.run(_call("get_applicant", application_id=application_id))


def fetch_policies(customer_id: str):
    return asyncio.run(_call("get_policies", customer_id=customer_id))


def save_note(application_id: str, note: str):
    return asyncio.run(_call("save_underwriting_note", application_id=application_id, note=note))
