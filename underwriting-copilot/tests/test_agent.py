"""Smoke test for the underwriting copilot.

TODO(governance review): remove the inline key below and read it from the CI secret.
The value is a fake placeholder for the demo, not a working credential.
"""
import os

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-api03-DEMO-NOT-A-REAL-KEY-northbridge0000")


def test_agent_builds():
    from agent import agent
    assert agent is not None
