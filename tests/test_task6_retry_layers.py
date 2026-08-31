from langchain_core.messages import ToolMessage

from langchain_learning_lab.task6_retry_layers import (
    run_model_retry_demo,
    run_tool_retry_demo,
)


def test_model_retry_succeeds_on_third_underlying_attempt() -> None:
    result, attempts = run_model_retry_demo()

    assert attempts == 3
    assert result["messages"][-1].content == "模型第三次尝试成功。"
    assert len(result["messages"]) == 2


def test_tool_retry_repeats_same_call_but_exposes_one_success_message() -> None:
    result, attempts = run_tool_retry_demo()

    assert attempts == 3
    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    assert "12" in str(tool_messages[0].content)
    assert result["messages"][-1].content == "SKU-1001 当前库存为12件。"
