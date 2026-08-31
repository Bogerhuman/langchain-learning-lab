"""Task 6: demonstrate model and tool retries with deterministic failures."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import tool


class FailThenSucceedModel(FakeMessagesListChatModel):
    """Raise transient model errors before returning a scripted response."""

    failures_before_success: int = 2
    attempts: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.attempts += 1
        if self.attempts <= self.failures_before_success:
            raise TimeoutError(f"temporary model timeout #{self.attempts}")
        return super()._generate(messages, stop, run_manager, **kwargs)


class ToolCallingFakeModel(FakeMessagesListChatModel):
    """Script tool selection while allowing create_agent to bind its schema."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any]],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        return self


_inventory_attempts = 0


def reset_inventory_attempts() -> None:
    global _inventory_attempts
    _inventory_attempts = 0


def inventory_attempts() -> int:
    return _inventory_attempts


@tool
def query_inventory(sku: str) -> dict[str, str | int]:
    """Return available inventory for one SKU from a read-only service."""
    global _inventory_attempts
    _inventory_attempts += 1
    if _inventory_attempts <= 2:
        raise TimeoutError(f"temporary inventory timeout #{_inventory_attempts}")
    return {"sku": sku, "available": 12, "warehouse": "SH-01"}


def run_model_retry_demo() -> tuple[dict[str, Any], int]:
    """Retry one model node internally without adding failed messages to state."""
    model = FailThenSucceedModel(
        responses=[AIMessage(content="模型第三次尝试成功。")]
    )
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[
            ModelRetryMiddleware(
                max_retries=2,
                retry_on=(TimeoutError,),
                on_failure="error",
                initial_delay=0,
                backoff_factor=0,
                jitter=False,
            )
        ],
        name="model_retry_demo",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
    return result, model.attempts


def run_tool_retry_demo() -> tuple[dict[str, Any], int]:
    """Retry the same read-only tool request before returning one ToolMessage."""
    reset_inventory_attempts()
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "query_inventory",
                        "args": {"sku": "SKU-1001"},
                        "id": "call_inventory",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="SKU-1001 当前库存为12件。"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[query_inventory],
        middleware=[
            ToolRetryMiddleware(
                max_retries=2,
                tools=["query_inventory"],
                retry_on=(TimeoutError,),
                on_failure="error",
                initial_delay=0,
                backoff_factor=0,
                jitter=False,
            )
        ],
        name="tool_retry_demo",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "查询SKU-1001库存"}]}
    )
    return result, inventory_attempts()


def print_message_summary(messages: list[BaseMessage]) -> None:
    for index, message in enumerate(messages, start=1):
        print(f"  [{index}] {message.type}: {message.content!r}")


def main() -> None:
    model_result, model_attempts = run_model_retry_demo()
    print("ModelRetryMiddleware:")
    print("  underlying model attempts:", model_attempts)
    print("  Agent messages:")
    print_message_summary(model_result["messages"])

    tool_result, tool_attempts = run_tool_retry_demo()
    print("\nToolRetryMiddleware:")
    print("  underlying tool attempts:", tool_attempts)
    tool_messages = [
        message
        for message in tool_result["messages"]
        if isinstance(message, ToolMessage)
    ]
    print("  ToolMessages visible to model:", len(tool_messages))
    print("  Agent messages:")
    print_message_summary(tool_result["messages"])


if __name__ == "__main__":
    main()
