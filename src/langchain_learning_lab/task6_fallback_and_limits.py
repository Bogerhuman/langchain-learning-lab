"""Task 6: demonstrate model fallback and semantic call limits offline."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import tool


class AlwaysFailModel(FakeMessagesListChatModel):
    """Represent an unavailable primary provider without making a network call."""

    attempts: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.attempts += 1
        raise TimeoutError("primary model provider timed out")


class CountingFakeModel(FakeMessagesListChatModel):
    """Count successful scripted model calls for teaching and assertions."""

    attempts: int = 0

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.attempts += 1
        return super()._generate(messages, stop, run_manager, **kwargs)


class ToolCallingCountingModel(CountingFakeModel):
    """Allow create_agent to bind tools to a deterministic scripted model."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any]],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        return self


_lookup_attempts = 0


def reset_lookup_attempts() -> None:
    global _lookup_attempts
    _lookup_attempts = 0


def lookup_attempts() -> int:
    return _lookup_attempts


@tool
def lookup_order(order_id: str) -> str:
    """Return the current status for one order."""
    global _lookup_attempts
    _lookup_attempts += 1
    return f"{order_id}: shipped"


def _tool_request(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "lookup_order",
                "args": {"order_id": "ORDER-1001"},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def run_model_fallback_demo() -> tuple[dict[str, Any], int, int]:
    """Use the fallback only after the primary model raises an exception."""
    primary = AlwaysFailModel(responses=[AIMessage(content="never returned")])
    fallback = CountingFakeModel(
        responses=[AIMessage(content="备用模型已接管并完成回答。")]
    )
    agent = create_agent(
        model=primary,
        tools=[],
        middleware=[ModelFallbackMiddleware(fallback)],
        name="model_fallback_demo",
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
    return result, primary.attempts, fallback.attempts


def run_model_call_limit_demo() -> tuple[dict[str, Any], int, int]:
    """Stop before a second model call even though the Agent loop needs one."""
    reset_lookup_attempts()
    model = ToolCallingCountingModel(
        responses=[
            _tool_request("call_model_limit"),
            AIMessage(content="这条回答不应生成。"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[lookup_order],
        middleware=[ModelCallLimitMiddleware(run_limit=1, exit_behavior="end")],
        name="model_call_limit_demo",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "查询订单并总结"}]}
    )
    return result, model.attempts, lookup_attempts()


def run_tool_call_limit_demo() -> tuple[dict[str, Any], int, int]:
    """Block a second tool call and let the model handle its error message."""
    reset_lookup_attempts()
    model = ToolCallingCountingModel(
        responses=[
            _tool_request("call_allowed"),
            _tool_request("call_blocked"),
            AIMessage(content="第二次查询被限制，我将使用第一次查询的结果。"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[lookup_order],
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="lookup_order",
                run_limit=1,
                exit_behavior="continue",
            )
        ],
        name="tool_call_limit_demo",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "连续查询两次订单状态"}]}
    )
    return result, model.attempts, lookup_attempts()


def print_message_summary(messages: list[BaseMessage]) -> None:
    for index, message in enumerate(messages, start=1):
        suffix = ""
        if isinstance(message, ToolMessage):
            suffix = f", status={message.status}, tool_call_id={message.tool_call_id}"
        print(f"  [{index}] {message.type}: {message.content!r}{suffix}")


def main() -> None:
    fallback_result, primary_calls, fallback_calls = run_model_fallback_demo()
    print("ModelFallbackMiddleware:")
    print("  primary calls:", primary_calls)
    print("  fallback calls:", fallback_calls)
    print_message_summary(fallback_result["messages"])

    model_limit_result, model_calls, tool_calls = run_model_call_limit_demo()
    print("\nModelCallLimitMiddleware(run_limit=1, end):")
    print("  actual model calls:", model_calls)
    print("  actual tool calls:", tool_calls)
    print_message_summary(model_limit_result["messages"])

    tool_limit_result, model_calls, tool_calls = run_tool_call_limit_demo()
    print("\nToolCallLimitMiddleware(run_limit=1, continue):")
    print("  actual model calls:", model_calls)
    print("  actual tool calls:", tool_calls)
    print_message_summary(tool_limit_result["messages"])


if __name__ == "__main__":
    main()
