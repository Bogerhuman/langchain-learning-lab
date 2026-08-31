from langchain_core.messages import ToolMessage

from langchain_learning_lab.task6_fallback_and_limits import (
    run_model_call_limit_demo,
    run_model_fallback_demo,
    run_tool_call_limit_demo,
)


def test_fallback_takes_over_after_primary_exception() -> None:
    result, primary_calls, fallback_calls = run_model_fallback_demo()

    assert primary_calls == 1
    assert fallback_calls == 1
    assert result["messages"][-1].content == "备用模型已接管并完成回答。"


def test_model_call_limit_stops_before_second_model_call() -> None:
    result, model_calls, tool_calls = run_model_call_limit_demo()

    assert model_calls == 1
    assert tool_calls == 1
    assert "limit" in str(result["messages"][-1].content).lower()


def test_tool_call_limit_blocks_second_call_and_returns_error_to_model() -> None:
    result, model_calls, tool_calls = run_tool_call_limit_demo()

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert model_calls == 3
    assert tool_calls == 1
    assert len(tool_messages) == 2
    assert tool_messages[0].status == "success"
    assert tool_messages[1].status == "error"
    assert tool_messages[1].tool_call_id == "call_blocked"
    assert "限制" in str(result["messages"][-1].content)
