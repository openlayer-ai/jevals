from jevals import Sample
from jevals._sample import as_sample, normalize_messages, normalize_tools


def test_openai_trace_derivations(sample):
    s = Sample(sample)
    assert s.input == "Where is my order? I'm jane@example.com"
    assert s.final_answer.startswith("Your order A123 shipped")
    assert s.system_prompt.startswith("You are a support agent")
    assert [tc.name for tc in s.tool_calls] == ["search_orders"]
    assert s.tool_calls[0].args == {"email": "jane@example.com"}
    assert "A123" in s.tool_calls[0].result
    assert len(s.tool_results) == 1 and s.tool_result == s.tool_results[0]
    assert s.tool_names == ["search_orders", "refund"]
    assert s.tool_call.name == "search_orders"  # last call by default
    assert [st["type"] for st in s.steps] == ["tool_call", "tool_result", "assistant"]
    assert s.has("messages") and not s.has("contexts")
    assert s.nonexistent is None


def test_explicit_keys_win():
    s = Sample(messages=[{"role": "assistant", "content": "derived"}], final_answer="explicit", input="q")
    assert s.final_answer == "explicit" and s.input == "q"


def test_single_turn_fields_build_messages():
    s = Sample(input="hi", output="hello", system_prompt="be nice")
    assert [m["role"] for m in s.messages] == ["system", "user", "assistant"]
    assert s.output == "hello" and s.final_answer == "hello"


def test_anthropic_blocks_are_split():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "weather in sf?"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t1", "name": "weather", "input": {"city": "sf"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": [{"type": "text", "text": "sunny 70F"}],
                }
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": "Sunny, 70F."}]},
    ]
    s = Sample(messages=msgs)
    assert s.input == "weather in sf?"
    assert (
        s.tool_calls[0].name == "weather"
        and s.tool_calls[0].args == {"city": "sf"}
        and s.tool_calls[0].result == "sunny 70F"
    )
    assert s.final_answer == "Sunny, 70F."
    roles = [m["role"] for m in s.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]


class _LC:
    def __init__(self, type_, content, tool_calls=None, tool_call_id=None):
        self.type, self.content, self.tool_calls, self.tool_call_id, self.name = (
            type_,
            content,
            tool_calls or [],
            tool_call_id,
            None,
        )


def test_langchain_like_messages():
    msgs = [
        _LC("human", "find A1"),
        _LC("ai", "", tool_calls=[{"id": "x", "name": "lookup", "args": {"id": "A1"}}]),
        _LC("tool", "found", tool_call_id="x"),
        _LC("ai", "Found it."),
    ]
    out = normalize_messages(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant"]
    s = Sample(messages=msgs)
    assert s.tool_calls[0].result == "found" and s.final_answer == "Found it."


def test_tool_result_matching_without_ids():
    msgs = [
        {"role": "user", "content": "x"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "a", "arguments": "{}"}},
                {"function": {"name": "b", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "content": "ra"},
        {"role": "tool", "content": "rb"},
    ]
    s = Sample(messages=msgs)
    assert [tc.result for tc in s.tool_calls] == ["ra", "rb"]


def test_tools_normalization():
    class T:
        name = "obj_tool"
        description = "does things"
        params_json_schema = {"type": "object"}

    out = normalize_tools(
        [
            {"name": "plain", "description": "d", "parameters": {}},
            {"type": "function", "function": {"name": "fn"}},
            "bare",
            T(),
            {"name": "anth", "input_schema": {"x": 1}},
        ]
    )
    assert [t["name"] for t in out] == ["plain", "fn", "bare", "obj_tool", "anth"]
    assert out[4]["parameters"] == {"x": 1}


def test_as_sample_shapes():
    assert as_sample("hello").output == "hello"
    assert as_sample([{"role": "user", "content": "q"}]).input == "q"
    s = Sample(a=1)
    assert as_sample(s) is s


def test_contexts_and_expected_aliases():
    s = Sample(retrieved_contexts=["c1", {"text": "c2"}], reference="ref")
    assert s.contexts[0] == "c1" and "c2" in s.contexts[1] and s.expected == "ref"
    assert Sample(context="one").contexts == ["one"]
