"""Characterization: the five tools and the entrypoint survive the transport refactor."""
import importlib.util
import os
import sys

BRIDGE = os.environ.get("ARBITER_BRIDGE", "/opt/arbiter/integrations/mcp/arbiter_mcp.py")
NAME = "arbiter_mcp_under_test"
TOOLS = ("arbiter_check", "arbiter_classify", "arbiter_score", "arbiter_gate", "arbiter_decide")


def _load():
    if NAME in sys.modules:                       # caching pattern: one exec per process
        return sys.modules[NAME]
    spec = importlib.util.spec_from_file_location(NAME, BRIDGE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
    return module


def test_the_five_tools_are_callable():
    module = _load()
    missing = [name for name in TOOLS if not callable(getattr(module, name, None))]
    assert missing == []


def test_main_is_the_entrypoint():
    assert callable(getattr(_load(), "main", None))
