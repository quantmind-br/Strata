"""Request guarantees supported by the native server (validated before SSE)."""
import math
from serve.frontend import tool_choice_of


class RequestError(ValueError):
    def __init__(self, param, message):
        super().__init__(f"{param}: {message}")
        self.param = param


def validate_sampling(req):
    ranges = {"temperature": (0, 2), "top_p": (0, 1), "min_p": (0, 1),
              "presence_penalty": (0, 2), "frequency_penalty": (0, 2),
              "repetition_penalty": (0, 1e6), "top_k": (1, 64),
              "seed": (0, 2**64 - 1), "penalty_last_n": (0, 2**31 - 1),
              "max_tokens": (-1, 2**31 - 1), "max_completion_tokens": (-1, 2**31 - 1)}
    integers = {"top_k", "seed", "penalty_last_n", "max_tokens", "max_completion_tokens"}
    for key, (low, high) in ranges.items():
        value = req.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)):
            raise RequestError(key, "expected a finite number")
        if key in integers and not isinstance(value, int):
            raise RequestError(key, "expected an integer")
        if value < low or (high is not None and value > high) or (key in ("top_p", "repetition_penalty") and value == 0):
            raise RequestError(key, f"value outside supported range {low}..{high}")
    tune = req.get("strata_tune")
    if tune is not None:
        if not isinstance(tune, dict):
            raise RequestError("strata_tune", "expected an object")
        for key, value in tune.items():
            if key not in ("pcie_frac", "spec_min_p") or isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)) or not 0 <= value <= 1:
                raise RequestError("strata_tune", "only pcie_frac and spec_min_p in 0..1 are supported")



def validate_request(req, api="openai"):
    if not isinstance(req, dict):
        raise RequestError("request", "expected an object")
    validate_sampling(req)
    for key in ("stream", "parallel_tool_calls"):
        if key in req and not isinstance(req[key], bool):
            raise RequestError(key, "expected a boolean")
    if "n" in req and (type(req["n"]) is not int or req["n"] != 1):
        raise RequestError("n", "only one completion is supported")
    tools = req.get("tools") or []
    if not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools):
        raise RequestError("tools", "expected an array of tool objects")
    mcp = req.get("strata_mcp") is True
    names = set()
    for tool in tools:
        fn = tool.get("function", tool)
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str):
            raise RequestError("tools", "each tool requires a function name")
        if fn.get("strict"):
            raise RequestError("tools", "strict schema decoding is unavailable")
        names.add(fn["name"])
    kind, name = tool_choice_of(req.get("tool_choice"))
    if kind == "unknown":
        raise RequestError("tool_choice", "expected auto, none, required or a named function")
    choice = req.get("tool_choice")
    if api == "anthropic" and isinstance(choice, dict) and "disable_parallel_tool_use" in choice:
        if not isinstance(choice["disable_parallel_tool_use"], bool):
            raise RequestError("tool_choice", "disable_parallel_tool_use must be a boolean")
    if kind in ("required", "named"):
        if not tools:
            raise RequestError("tool_choice", "forcing a tool call requires the request's own tools")
        if mcp:
            raise RequestError("tool_choice", "forcing a tool call is unsupported with MCP tools")
        if kind == "named" and name not in names:
            raise RequestError("tool_choice", "names a tool the request does not offer")
    if kind == "none" and mcp:
        raise RequestError("tool_choice", "none is unsupported with MCP tools")
    if mcp and single_call(req, api):
        raise RequestError("parallel_tool_calls", "a single-call limit is unsupported with MCP tools")
    if req.get("logprobs") or req.get("top_logprobs") or req.get("logit_bias"):
        raise RequestError("logprobs/logit_bias", "unsupported by this engine")
    if not isinstance(req.get("messages"), list) or not req["messages"] or any(not isinstance(m, dict) for m in req["messages"]):
        raise RequestError("messages", "expected a nonempty array of messages")


def single_call(req, api="openai"):
    """True when at most one tool call may be returned (OpenAI parallel_tool_calls, Anthropic disable_parallel_tool_use)."""
    choice = req.get("tool_choice")
    return req.get("parallel_tool_calls") is False or (
        api == "anthropic" and isinstance(choice, dict) and choice.get("disable_parallel_tool_use") is True)
