from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from langchain_learning_lab.task6_middleware_lifecycle import (
    LifecycleLoggingMiddleware,
    create_observed_agent,
    event_counts,
)


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any]],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        return self


def test_middleware_hooks_follow_agent_and_model_lifecycle() -> None:
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculate_order_total",
                        "args": {
                            "unit_price_yuan": 10,
                            "quantity": 2,
                            "discount_percent": 0,
                        },
                        "id": "call_test",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="最终金额为20元。"),
        ]
    )
    middleware = LifecycleLoggingMiddleware()
    agent = create_observed_agent(middleware, model=model)

    result = agent.invoke({"messages": [{"role": "user", "content": "算价格"}]})

    assert result["messages"][-1].content == "最终金额为20元。"
    assert middleware.events == [
        "before_agent",
        "before_model",
        "wrap_model_call:enter",
        "wrap_model_call:exit",
        "after_model",
        "before_model",
        "wrap_model_call:enter",
        "wrap_model_call:exit",
        "after_model",
        "after_agent",
    ]


def test_agent_hooks_run_once_and_model_hooks_run_per_model_call() -> None:
    events = [
        "before_agent",
        "before_model",
        "wrap_model_call:enter",
        "wrap_model_call:exit",
        "after_model",
        "before_model",
        "wrap_model_call:enter",
        "wrap_model_call:exit",
        "after_model",
        "after_agent",
    ]

    counts = event_counts(events)

    assert counts["before_agent"] == 1
    assert counts["after_agent"] == 1
    assert counts["before_model"] == 2
    assert counts["after_model"] == 2
    assert counts["wrap_model_call:enter"] == 2
    assert counts["wrap_model_call:exit"] == 2
