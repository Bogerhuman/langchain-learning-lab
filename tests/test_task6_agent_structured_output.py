from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable

from langchain_learning_lab.task6_agent_structured_output import create_ticket_agent
from langchain_learning_lab.task6_structured_output import SupportTicket


VALID_ARGS = {
    "category": "database",
    "priority": "critical",
    "summary": "支付服务因数据库连接池耗尽而不可用",
    "affected_services": ["payment-api"],
    "requires_human": True,
    "confidence": 0.95,
}


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Deterministic fake that accepts Agent tool binding without network."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any]],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        return self


def structured_call(args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "SupportTicket",
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def test_agent_places_validated_object_in_structured_response() -> None:
    model = ToolCallingFakeModel(
        responses=[structured_call(VALID_ARGS, "call_valid")]
    )
    agent = create_ticket_agent(model)

    result = agent.invoke({"messages": [{"role": "user", "content": "故障"}]})

    assert isinstance(result["structured_response"], SupportTicket)
    assert result["structured_response"].priority == "critical"
    assert any(
        isinstance(message, ToolMessage)
        and message.content == "Structured support ticket accepted."
        for message in result["messages"]
    )


def test_tool_strategy_returns_validation_error_to_model_then_retries() -> None:
    invalid_args = {**VALID_ARGS, "confidence": 1.5}
    model = ToolCallingFakeModel(
        responses=[
            structured_call(invalid_args, "call_invalid"),
            structured_call(VALID_ARGS, "call_retry"),
        ]
    )
    agent = create_ticket_agent(model)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "故障"}]},
        config={"recursion_limit": 6},
    )

    assert result["structured_response"].confidence == 0.95
    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    assert "Error" in str(tool_messages[0].content)
    assert tool_messages[1].content == "Structured support ticket accepted."
