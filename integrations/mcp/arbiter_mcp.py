#!/usr/bin/env python3
"""An MCP server that gives a coding agent a System One: five fast, typed decisions.

Agents are good at deciding things and slow at it. Every "is this command safe", "which of these
passages matters", "how risky is this change" costs a full model turn, and the agent pays for it
in latency and in context. These tools answer the same questions against a local arbiter server in
tens of milliseconds, return a probability rather than a paragraph, and leave the agent's
context alone.

Tools:

    arbiter_classify   pick one of up to 12 labelled options
    arbiter_score      place something on an ordered scale of 2-10 levels
    arbiter_check      the probability that a statement about the state is true
    arbiter_gate       is this action safe to run: allow | confirm | block, with the signals
    arbiter_decide     any set of questions at once, raw -- one forward pass for all of them

Every tool returns structured content (the probabilities) plus one short line of text, so an
agent can act on the numbers and a human reading the transcript can see why.

Run it:  ARBITER_URL=http://localhost:8010 python arbiter_mcp.py
Install: pip install "mcp>=2"   (or `pip install .` in this directory for the `arbiter-mcp` script)

Configuration comes from the environment: ARBITER_URL (default http://localhost:8010),
ARBITER_API_KEY (optional), ARBITER_MODEL (default "auto" -- let the server route).

The transport is chosen by the command line, then the environment, then the default stdio:

    --transport stdio | sse | streamable-http   (env: ARBITER_MCP_TRANSPORT)
    --host HOST --port PORT --path PATH         (env: ARBITER_MCP_HOST / _PORT / _PATH)
    --message-path PATH                          (env: ARBITER_MCP_MESSAGE_PATH; sse only)

stdio stays the default so an agent that spawns this file as a subprocess sees no change.
The HTTP transports expose the same five tools to any MCP client on the wire, which is what
lets the bridge run next to the server it fronts instead of on every agent host. Defaults
bind 127.0.0.1:8020 on /mcp; serving the LAN is a deliberate act, so set --host 0.0.0.0.
"""
import argparse
import importlib.util
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

try:
    from mcp.server.mcpserver import MCPServer
except ModuleNotFoundError as exc:                       # pragma: no cover - install-time path
    raise SystemExit("arbiter-mcp needs the official MCP SDK v2 or newer: pip install 'mcp>=2'") from exc

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult, TextContent, ToolAnnotations

ARBITER_URL = os.environ.get("ARBITER_URL", "http://localhost:8010").rstrip("/")
ARBITER_API_KEY = os.environ.get("ARBITER_API_KEY")
ARBITER_MODEL = os.environ.get("ARBITER_MODEL", "auto")
TIMEOUT = float(os.environ.get("ARBITER_TIMEOUT", "30"))

# Transport selection. The command line wins, the environment is the second layer, and stdio is
# the default, so an agent that spawns this file as a subprocess is unaffected by the new
# transports. The names of the per-transport keyword arguments are the SDK's own
# (run_streamable_http_async takes host/port/streamable_http_path, run_sse_async takes
# host/port/sse_path/message_path) and the test suite pins them, because a rename upstream would
# otherwise fail at the first request rather than at import.
TRANSPORTS = ("stdio", "sse", "streamable-http")
DEFAULT_TRANSPORT = os.environ.get("ARBITER_MCP_TRANSPORT", "stdio")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8020
DEFAULT_PATH = "/mcp"
DEFAULT_MESSAGE_PATH = "/messages/"

# Option descriptions share a fixed token budget with the state, and accuracy falls off well
# before the server's hard limit, so the tool refuses long option lists rather than answering
# badly. The fix is hierarchical: a coarse classify, then a second one inside the winner.
MAX_OPTIONS = 12
MIN_LEVELS, MAX_LEVELS = 2, 10

# The gate's questions, weights, cut lines and text rules live in one place, next to the Claude
# Code hook, so the hook, this tool and examples/tool_call_guard.py cannot drift apart. It is
# loaded by path rather than imported as a package because this file is also installed on its
# own; ARBITER_GUARD_POLICY points at the copy when it has been moved somewhere else.
_POLICY_PATH = os.environ.get("ARBITER_GUARD_POLICY") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "claude-code", "hooks", "guard_policy.py")
if not os.path.exists(_POLICY_PATH):
    raise SystemExit("arbiter-mcp needs the guard policy at %s. Copy "
                     "integrations/claude-code/hooks/guard_policy.py next to this file's "
                     "sibling directory, or set ARBITER_GUARD_POLICY." % _POLICY_PATH)
_POLICY_SPEC = importlib.util.spec_from_file_location("guard_policy", _POLICY_PATH)
policy = importlib.util.module_from_spec(_POLICY_SPEC)
_POLICY_SPEC.loader.exec_module(policy)

# The hook speaks Claude Code's vocabulary; an agent calling this tool gets the MCP one.
GATE_WORD = {"allow": "allow", "ask": "confirm", "deny": "block"}

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True,
                            openWorldHint=True)

server = MCPServer(
    name="arbiter",
    instructions=("Fast typed decisions from a local System One server. Prefer these tools "
                  "over reasoning about a routine judgement yourself: they answer in tens of "
                  "milliseconds and give you a probability you can threshold. Ask every question "
                  "need in one arbiter_decide call -- they all run in a single forward pass, so ten "
                  "questions cost what one costs. Never treat a probability as a verdict: pick "
                  "your own thresholds and leave a middle band where you ask the user."),
)


# --------------------------------------------------------------------------- transport

def call_arbiter(state: Any, questions: Dict[str, Any], model: Optional[str] = None) -> Dict[str, Any]:
    body = json.dumps({"state": state, "questions": questions,
                       "model": model or ARBITER_MODEL}).encode()
    headers = {"content-type": "application/json"}
    if ARBITER_API_KEY:
        headers["authorization"] = "Bearer %s" % ARBITER_API_KEY
    request = urllib.request.Request(ARBITER_URL + "/v1/systemone", data=body, headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail)["error"]["message"]
        except Exception:
            pass
        raise ToolError("the arbiter server rejected the request (HTTP %d): %s" % (exc.code, detail))
    except OSError as exc:
        raise ToolError("cannot reach the arbiter server at %s (%s). Start it, or set ARBITER_URL."
                         % (ARBITER_URL, exc))


def result(text: str, structured: Dict[str, Any]) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)],
                          structured_content=structured)


def meta(response: Dict[str, Any]) -> Dict[str, Any]:
    return {"checkpoint": response.get("routing", {}).get("model"),
            "latency_ms": response.get("latency_ms")}


# --------------------------------------------------------------------------- validation

def require_text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ToolError("%s must be a non-empty string, got %r" % (field, value))


def require_state(state: Any) -> None:
    if state is None or (isinstance(state, str) and not state.strip()):
        raise ToolError("state must be a non-empty string, object or array: it is the thing "
                         "being judged, and an empty one has no answer")


def check_options(options: Any) -> Dict[str, Optional[str]]:
    if not isinstance(options, dict) or not options:
        raise ToolError("options must be an object mapping option name -> description")
    if len(options) < 2:
        raise ToolError("a choice needs at least 2 options, got %d" % len(options))
    if len(options) > MAX_OPTIONS:
        raise ToolError(
            "a choice takes at most %d options here, got %d. Option descriptions share a fixed "
            "token budget with the state, so long lists get less accurate, not just slower. Ask "
            "a coarse question first and a second, narrower one inside the winning group."
            % (MAX_OPTIONS, len(options)))
    for name, description in options.items():
        if description is not None and not isinstance(description, str):
            raise ToolError("option %r: the description must be a string or null" % name)
    return options


def check_levels(levels: Any) -> List[str]:
    if not isinstance(levels, list) or not all(isinstance(x, str) and x.strip() for x in levels):
        raise ToolError("levels must be an array of non-empty level descriptions, lowest first")
    if not MIN_LEVELS <= len(levels) <= MAX_LEVELS:
        raise ToolError("a score needs between %d and %d levels, got %d"
                         % (MIN_LEVELS, MAX_LEVELS, len(levels)))
    return levels


def check_questions(questions: Any) -> Dict[str, Any]:
    if not isinstance(questions, dict) or not questions:
        raise ToolError("questions must be a non-empty object of question id -> question")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in ("noul", "choice", "score"):
            raise ToolError("question %r: type must be one of noul, choice, score" % qid)
        require_text(q.get("instructions"), "question %r: instructions" % qid)
        if q["type"] == "choice":
            check_options(q.get("criteria"))
        elif q["type"] == "score":
            check_levels(q.get("criteria"))
        elif q.get("criteria") is not None:
            extra = sorted(set(q["criteria"]) - {"true", "false"})
            if extra:
                raise ToolError("question %r: noul criteria may only have 'true' and 'false', "
                                 "found %s" % (qid, extra))
    return questions


# --------------------------------------------------------------------------- tools

@server.tool(
    title="Classify",
    description=("Pick one of up to 12 labelled options for a piece of state, with the full "
                 "probability distribution. Use it for routing, intent, category and triage "
                 "decisions instead of reasoning about them yourself."),
    annotations=READ_ONLY, structured_output=False)
def arbiter_classify(state: Any, instructions: str, options: Dict[str, Optional[str]],
                  model: Optional[str] = None) -> CallToolResult:
    require_state(state)
    require_text(instructions, "instructions")
    check_options(options)
    response = call_arbiter(state, {"q": {"type": "choice", "instructions": instructions,
                                       "criteria": options}}, model)
    answer = response["answers"]["q"]
    payload = {"choice": answer["choice"], "probabilities": answer["probabilities"],
               "confidence": answer["confidence"]}
    payload.update(meta(response))
    return result("%s (p=%.2f, confidence %.2f)" % (
        answer["choice"], answer["probabilities"][answer["choice"]], answer["confidence"]), payload)


@server.tool(
    title="Score",
    description=("Place a piece of state on an ordered scale of 2-10 levels, lowest first. "
                 "Returns the expected level as a float plus the distribution. Ordinal scales "
                 "are the weakest of the three primitives: where a hard boundary matters, phrase "
                 "the boundary as a arbiter_check instead."),
    annotations=READ_ONLY, structured_output=False)
def arbiter_score(state: Any, instructions: str, levels: List[str],
               model: Optional[str] = None) -> CallToolResult:
    require_state(state)
    require_text(instructions, "instructions")
    check_levels(levels)
    response = call_arbiter(state, {"q": {"type": "score", "instructions": instructions,
                                       "criteria": levels}}, model)
    answer = response["answers"]["q"]
    payload = {"score": answer["score"], "max_score": len(levels) - 1, "legend": answer["legend"],
               "probabilities": answer["probabilities"], "confidence": answer["confidence"]}
    payload.update(meta(response))
    nearest = answer["legend"][str(int(round(answer["score"])))]
    return result("%.2f / %d (%s), confidence %.2f"
                  % (answer["score"], len(levels) - 1, nearest, answer["confidence"]), payload)


@server.tool(
    title="Check",
    description=("The probability that a statement about the state is true. The strongest of the "
                 "three primitives: phrase the thing you actually want to know as a statement "
                 "('this passage answers the question', 'this change needs a migration') and "
                 "threshold the number yourself."),
    annotations=READ_ONLY, structured_output=False)
def arbiter_check(state: Any, instructions: str, true_desc: Optional[str] = None,
               false_desc: Optional[str] = None, model: Optional[str] = None) -> CallToolResult:
    require_state(state)
    require_text(instructions, "instructions")
    criteria = {k: v for k, v in (("true", true_desc), ("false", false_desc)) if v}
    question: Dict[str, Any] = {"type": "noul", "instructions": instructions}
    if criteria:
        question["criteria"] = criteria
    response = call_arbiter(state, {"q": question}, model)
    answer = response["answers"]["q"]
    payload = {"probability": answer["noul"], "confidence": answer["confidence"]}
    payload.update(meta(response))
    return result("%.2f (%s)" % (answer["noul"], "likely true" if answer["noul"] >= 0.5
                                 else "likely false"), payload)


@server.tool(
    title="Gate an action",
    description=("Judge whether a shell command is safe to run. Returns allow | confirm | block "
                 "with the risk score and the signals behind it. Call it before anything "
                 "destructive; it costs tens of milliseconds, and read-only commands are "
                 "answered without a round trip at all."),
    annotations=READ_ONLY, structured_output=False)
def arbiter_gate(action: str, context: Optional[str] = None, cwd: Optional[str] = None,
              model: Optional[str] = None) -> CallToolResult:
    require_text(action, "action")
    if policy.is_read_only(action):
        decision, risk, reason = policy.decide(None, action)
        return result("%s: %s" % (GATE_WORD[decision], reason),
                      {"recommendation": GATE_WORD[decision], "risk": risk, "signals": {},
                       "reason": reason, "checkpoint": None, "latency_ms": 0.0})
    state = policy.build_state(action, context, cwd)
    response = call_arbiter(state, policy.QUESTIONS, model)
    decision, risk, reason = policy.decide(response["answers"], action)
    signals = {qid: round(float(a["noul"] if a["type"] == "noul" else a["score"]), 4)
               for qid, a in response["answers"].items()}
    payload = {"recommendation": GATE_WORD[decision], "risk": round(risk, 4),
               "signals": signals, "reason": reason}
    payload.update(meta(response))
    return result("%s: %s" % (GATE_WORD[decision], reason), payload)


@server.tool(
    title="Decide (batched)",
    description=("Ask any number of questions about one state in a single forward pass. This is "
                 "the tool to reach for when you have more than one thing to know: ten questions "
                 "cost about what one costs. `questions` maps an id to "
                 "{type: noul|choice|score, instructions, criteria}."),
    annotations=READ_ONLY, structured_output=False)
def arbiter_decide(state: Any, questions: Dict[str, Any],
                model: Optional[str] = None) -> CallToolResult:
    require_state(state)
    check_questions(questions)
    response = call_arbiter(state, questions, model)
    summary = []
    for qid, a in response["answers"].items():
        if a["type"] == "noul":
            summary.append("%s=%.2f" % (qid, a["noul"]))
        elif a["type"] == "choice":
            summary.append("%s=%s" % (qid, a["choice"]))
        else:
            summary.append("%s=%.2f" % (qid, a["score"]))
    payload = {"answers": response["answers"], "usage": response.get("usage", {})}
    payload.update(meta(response))
    return result(", ".join(summary), payload)


def transport_plan(args: argparse.Namespace) -> Tuple[str, Dict[str, Any]]:
    """Decide how to serve, without serving. Pure: returns (transport, kwargs) the caller runs with.

    Kept apart from main() because run() never returns -- each SDK transport ends in
    anyio.run(...) and blocks until the process is stopped. A selector that also ran would be
    untestable; this one is a function of its arguments, so the suite can pin the flag and
    environment precedence and the exact SDK keyword names without starting a socket.
    """
    if args.transport == "stdio":
        return "stdio", {}                       # stdio takes no host or port; the agent owns the pipes
    kwargs: Dict[str, Any] = {"host": args.host, "port": args.port}
    if args.transport == "streamable-http":
        kwargs["streamable_http_path"] = args.path
    else:                                        # sse: the legacy pair of endpoints
        kwargs["sse_path"] = args.path
        kwargs["message_path"] = args.message_path
    return args.transport, kwargs


def parse_args(argv: "list[str] | None" = None) -> argparse.Namespace:
    """Flags over environment over defaults, as in core(1). Every knob has an ARBITER_MCP_* twin.

    The environment is read here, at call time, and not captured at import: the module is
    imported once by the test suite and by anyone embedding it, and a daemon that froze its
    configuration at import would ignore the unit file it is later handed.
    """
    parser = argparse.ArgumentParser(prog="arbiter-mcp",
                                     description="Serve the five arbiter decisions as MCP tools.")
    parser.add_argument(
        "--transport", choices=TRANSPORTS,
        default=os.environ.get("ARBITER_MCP_TRANSPORT", DEFAULT_TRANSPORT),
        help="MCP wire transport (env: ARBITER_MCP_TRANSPORT; default stdio)")
    parser.add_argument(
        "--host", default=os.environ.get("ARBITER_MCP_HOST", DEFAULT_HOST),
        help="bind address for the HTTP transports (env: ARBITER_MCP_HOST; default 127.0.0.1; "
             "use 0.0.0.0 to serve the LAN)")
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("ARBITER_MCP_PORT", DEFAULT_PORT)),
        help="bind port for the HTTP transports (env: ARBITER_MCP_PORT; default 8020)")
    parser.add_argument(
        "--path", default=os.environ.get("ARBITER_MCP_PATH", DEFAULT_PATH),
        help="the MCP endpoint path (env: ARBITER_MCP_PATH; default /mcp)")
    parser.add_argument(
        "--message-path", default=os.environ.get("ARBITER_MCP_MESSAGE_PATH", DEFAULT_MESSAGE_PATH),
        help="POST endpoint for the sse transport (env: ARBITER_MCP_MESSAGE_PATH; default /messages/)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    transport, kwargs = transport_plan(args)
    server.run(transport=transport, **kwargs)


if __name__ == "__main__":
    main()
