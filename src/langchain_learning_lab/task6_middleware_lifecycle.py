"""Task 6: observe Agent lifecycle hooks with custom middleware."""

from collections import Counter
from collections.abc import Callable
from time import perf_counter
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
)
from langgraph.runtime import Runtime

from langchain_learning_lab.task3_manual_tool_loop import (
    calculate_order_total,
    create_model,
)


class LifecycleLoggingMiddleware(AgentMiddleware):
    """Record deterministic Agent/model hooks for one teaching process."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []
        self.model_call_durations_ms: list[float] = []

    def before_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        self.events.append("before_agent")
        return None

    def before_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        self.events.append("before_model")
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self.events.append("wrap_model_call:enter")
        started_at = perf_counter()
        try:
            return handler(request)
        finally:
            elapsed_ms = (perf_counter() - started_at) * 1000
            self.model_call_durations_ms.append(elapsed_ms)
            self.events.append("wrap_model_call:exit")

    def after_model(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        self.events.append("after_model")
        return None

    def after_agent(
        self,
        state: AgentState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        self.events.append("after_agent")
        return None


def create_observed_agent(middleware: LifecycleLoggingMiddleware, model=None):
    """Build a tool-using Agent whose lifecycle can be observed externally."""
    return create_agent(
        model=model or create_model(),
        tools=[calculate_order_total],
        middleware=[middleware],
        system_prompt=(
            "你是订单助手。所有金额计算必须调用工具，读取工具结果后用中文回答。"
        ),
        name="observed_order_agent",
    )


def event_counts(events: list[str]) -> Counter:
    """Count hooks without losing their original ordered event list."""
    return Counter(events)


def main() -> None:
    middleware = LifecycleLoggingMiddleware()
    agent = create_observed_agent(middleware)
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "商品单价79.9元，购买3件，打85折，最终多少钱？",
                }
            ]
        },
        config={"recursion_limit": 10},
    )

    print("Middleware event order:")
    for index, event in enumerate(middleware.events, start=1):
        print(f"  {index}. {event}")

    print("\nHook counts:")
    for event, count in event_counts(middleware.events).items():
        print(f"  {event}: {count}")

    print("\nModel call durations:")
    for index, duration_ms in enumerate(
        middleware.model_call_durations_ms,
        start=1,
    ):
        print(f"  call {index}: {duration_ms:.2f} ms")

    print("\nFinal answer:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
