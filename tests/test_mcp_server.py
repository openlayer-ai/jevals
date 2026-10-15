import asyncio

import pytest

pytest.importorskip("mcp")

from jevals.integrations.mcp_server import build_server  # noqa: E402


def test_mcp_server_builds_and_lists_evals():
    # mcp 2.x renamed FastMCP to MCPServer, and list_evals used to build every eval with no
    # arguments, which CustomRubric doesn't allow.
    server = build_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"list_evals", "describe_eval", "evaluate", "gate", "docs"} <= names
    assert "custom_rubric" in str(asyncio.run(server.call_tool("list_evals", {})))


def test_contract_tools(tmp_path):
    server = build_server()
    names = {t.name for t in asyncio.run(server.list_tools())}
    assert {"contract_rules", "contract_check"} <= names
    out = tmp_path / "contract.yaml"
    args = {
        "prompt": "- Ask for the order number.\n- Reply in the customer's language.\n",
        "out": str(out),
        "use_model": False,
        "backend": "mock",
    }
    res = str(asyncio.run(server.call_tool("contract_rules", args)))
    assert "Ask for the order number." in res and out.exists()
    ev = tmp_path / "e.yaml"
    ev.write_text("name: e\nquestions:\n  a: {type: noul, instructions: 'Asks?'}\n")
    args = {"contract": str(out), "evals": [str(ev)], "use_model": False, "backend": "mock"}
    res = str(asyncio.run(server.call_tool("contract_check", args)))
    assert "uncovered" in res and "Ask for the order number." in res
