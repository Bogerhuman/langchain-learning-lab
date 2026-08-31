from langchain_core.messages import ToolMessage

from langchain_learning_lab.task6_human_in_the_loop import run_decision_demo


def test_sensitive_tool_is_not_executed_while_agent_is_paused() -> None:
    result = run_decision_demo("approve")

    action = result.interrupt["action_requests"][0]
    assert action["name"] == "cancel_order"
    assert action["args"] == {"order_id": "ORDER-1001"}
    assert result.tool_calls_before_resume == []


def test_approve_executes_original_action() -> None:
    result = run_decision_demo("approve")

    assert result.tool_calls_after_resume == ["ORDER-1001"]
    assert result.messages[-1].content == "审批通过，订单取消操作已完成。"


def test_edit_executes_only_edited_arguments() -> None:
    result = run_decision_demo("edit")

    assert result.tool_calls_after_resume == ["ORDER-2002"]
    tool_message = next(
        message for message in result.messages if isinstance(message, ToolMessage)
    )
    assert tool_message.status == "success"
    assert tool_message.tool_call_id == "call_cancel_order"
    assert "ORDER-2002" in str(tool_message.content)


def test_reject_skips_tool_and_returns_error_feedback() -> None:
    result = run_decision_demo("reject")

    assert result.tool_calls_after_resume == []
    tool_message = next(
        message for message in result.messages if isinstance(message, ToolMessage)
    )
    assert tool_message.status == "error"
    assert tool_message.tool_call_id == "call_cancel_order"
    assert "不允许取消" in str(tool_message.content)
    assert "没有被取消" in str(result.messages[-1].content)
