"""Transport selection: flags beat the environment, the environment beats the defaults."""
import importlib.util
import os
import sys

import pytest

BRIDGE = os.environ.get("ARBITER_BRIDGE", "/opt/arbiter/integrations/mcp/arbiter_mcp.py")
NAME = "arbiter_mcp_under_test"
ENV_VARS = ("ARBITER_MCP_TRANSPORT", "ARBITER_MCP_HOST", "ARBITER_MCP_PORT", "ARBITER_MCP_PATH")


def _load():
    if NAME in sys.modules:
        return sys.modules[NAME]
    spec = importlib.util.spec_from_file_location(NAME, BRIDGE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
    return module


def _clean_env():
    for key in ENV_VARS:
        os.environ.pop(key, None)


def test_default_is_stdio_and_carries_no_kwargs():
    _clean_env()
    module = _load()
    assert module.transport_plan(module.parse_args([])) == ("stdio", {})


def test_streamable_http_defaults_match_the_measured_sdk_signature():
    _clean_env()
    module = _load()
    plan = module.transport_plan(module.parse_args(["--transport", "streamable-http"]))
    assert plan == ("streamable-http", {"host": "127.0.0.1", "port": 8020,
                                        "streamable_http_path": "/mcp"})


def test_flags_select_streamable_http_and_build_its_kwargs():
    _clean_env()
    module = _load()
    plan = module.transport_plan(module.parse_args([
        "--transport", "streamable-http", "--host", "10.0.0.1",
        "--port", "9001", "--path", "/mcp"]))
    assert plan == ("streamable-http", {"host": "10.0.0.1", "port": 9001,
                                        "streamable_http_path": "/mcp"})


def test_sse_uses_the_measured_member_names():
    _clean_env()
    module = _load()
    transport, kwargs = module.transport_plan(module.parse_args(["--transport", "sse"]))
    assert transport == "sse"
    assert set(kwargs) == {"host", "port", "sse_path", "message_path"}
    assert kwargs["sse_path"] == "/mcp"


def test_environment_is_the_second_layer_and_flags_win():
    module = _load()
    _clean_env()
    os.environ.update({"ARBITER_MCP_TRANSPORT": "streamable-http",
                       "ARBITER_MCP_HOST": "127.0.0.1",
                       "ARBITER_MCP_PORT": "8123",
                       "ARBITER_MCP_PATH": "/mcp"})
    try:
        assert module.transport_plan(module.parse_args([])) == (
            "streamable-http", {"host": "127.0.0.1", "port": 8123,
                               "streamable_http_path": "/mcp"})
        plan = module.transport_plan(module.parse_args(["--host", "192.168.0.7", "--port", "9999"]))
        assert plan[1]["host"] == "192.168.0.7"
        assert plan[1]["port"] == 9999
    finally:
        _clean_env()


def test_an_unknown_transport_is_rejected_like_a_well_behaved_cli():
    _clean_env()
    module = _load()
    with pytest.raises(SystemExit) as raised:
        module.parse_args(["--transport", "carrier-pigeon"])
    assert raised.value.code == 2
