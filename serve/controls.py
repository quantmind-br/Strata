"""Request guarantees supported by the native server (validated before SSE)."""
import math


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


def stop_sequences(req):
    stops = req.get("stop", req.get("stop_sequences"))
    if stops is None:
        return []
    if isinstance(stops, str):
        stops = [stops]
    if not isinstance(stops, list) or not 1 <= len(stops) <= 4 or any(not isinstance(s, str) or not s for s in stops):
        raise RequestError("stop", "expected a nonempty string or one to four nonempty strings")
    return stops


def validate_request(req, api="openai"):
    if not isinstance(req, dict):
        raise RequestError("request", "expected an object")
    validate_sampling(req)
    stop_sequences(req)
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
    choice = tool_choice(req, api)
    if choice == "none" and (tools or mcp):
        raise RequestError("tool_choice", "none is unsupported when tools are offered; omit the tools instead")
    if choice not in (None, "none"):
        if not tools:
            raise RequestError("tool_choice", "forcing a tool call requires the request's own tools")
        if mcp:
            raise RequestError("tool_choice", "forcing a tool call is unsupported with MCP tools")
        if isinstance(choice, tuple) and choice[1] not in names:
            raise RequestError("tool_choice", "names a tool the request does not offer")
    if mcp and single_call(req, api):
        raise RequestError("parallel_tool_calls", "a single-call limit is unsupported with MCP tools")
    if req.get("logprobs") or req.get("top_logprobs") or req.get("logit_bias"):
        raise RequestError("logprobs/logit_bias", "unsupported by this engine")
    if not isinstance(req.get("messages"), list) or not req["messages"] or any(not isinstance(m, dict) for m in req["messages"]):
        raise RequestError("messages", "expected a nonempty array of messages")


def tool_choice(req, api="openai"):
    """The request's tool choice -> None (automatic), "none", "required" or ("function", name)."""
    choice = req.get("tool_choice")
    if choice in (None, "auto", "none", "required"):
        return None if choice == "auto" else choice
    if isinstance(choice, dict):
        kind = choice.get("type")
        if api == "anthropic":
            if kind == "auto":
                return None
            if kind in ("any", "none"):
                return {"any": "required"}.get(kind, kind)
            if kind == "tool" and isinstance(choice.get("name"), str):
                return ("function", choice["name"])
        elif kind == "function" and isinstance((choice.get("function") or {}).get("name"), str):
            return ("function", choice["function"]["name"])
    raise RequestError("tool_choice", "expected auto, none, required or a named function")


def single_call(req, api="openai"):
    """True when at most one tool call may be returned (OpenAI parallel_tool_calls, Anthropic disable_parallel_tool_use)."""
    choice = req.get("tool_choice")
    return req.get("parallel_tool_calls") is False or (
        api == "anthropic" and isinstance(choice, dict) and choice.get("disable_parallel_tool_use") is True)


CALL_PREFIX = "<tool_call>\n<function="


def tool_policy(req, api="openai"):
    """-> (prefix, max_calls) for a validated request.  prefix: the template text a forced choice starts the answer
    with (a call to the named function - the only one offered, for "required" - or else a call whose name the model
    writes, which must then be an offered tool); max_calls: 1 under a single-call
    limit, else None.  The server returns only complete calls, so the limit is exact; a forced call that does not
    complete is an error, never plain text."""
    choice = tool_choice(req, api)
    names = [t.get("function", t).get("name") for t in req.get("tools") or []]
    if choice == "required" and len(names) == 1:
        choice = ("function", names[0])          # one offered tool: requiring a call is calling that tool
    prefix = ""
    if choice == "required":
        prefix = CALL_PREFIX
    elif isinstance(choice, tuple):
        prefix = CALL_PREFIX + choice[1] + ">\n"
    return prefix, (1 if single_call(req, api) else None)


class StopFilter:
    """Hold only a possible delimiter prefix; never emit part of a stop sequence."""
    def __init__(self, stops):
        self.stops, self.pending, self.matched = stops, "", None

    def feed(self, text):
        if self.matched is not None:
            return ""
        text = self.pending + text
        matches = [(text.find(s), s) for s in self.stops if s in text]
        if matches:
            pos, self.matched = min(matches, key=lambda pair: pair[0])
            self.pending = ""
            return text[:pos]
        hold = max((n for s in self.stops for n in range(1, min(len(s), len(text) + 1)) if text.endswith(s[:n])), default=0)
        self.pending = text[-hold:] if hold else ""
        return text[:-hold] if hold else text

    def finish(self):
        text, self.pending = self.pending, ""
        return text
