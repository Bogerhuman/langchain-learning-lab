from langchain_core.messages import AIMessage, ToolMessage

from langchain_learning_lab.task4_streaming import describe_update


def test_describe_model_update_includes_tool_call() -> None:
    message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculate_order_total",
                "args": {"unit_price_yuan": 10, "quantity": 2},
                "id": "call_123",
                "type": "tool_call",
            }
        ],
    )

    lines = describe_update("model", [message])

    assert lines[0] == "Node: model"
    assert any("calculate_order_total" in line for line in lines)
    assert any("call_123" in line for line in lines)


def test_describe_tools_update_includes_matching_result_id() -> None:
    message = ToolMessage(
        content='{"total_yuan": 20}',
        name="calculate_order_total",
        tool_call_id="call_123",
    )

    lines = describe_update("tools", [message])

    assert lines[0] == "Node: tools"
    assert any("call_123" in line for line in lines)
    assert any("total_yuan" in line for line in lines)
