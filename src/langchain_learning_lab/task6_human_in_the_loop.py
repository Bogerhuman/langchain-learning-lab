"""Task 6: pause sensitive tool calls for deterministic human review."""

from dataclasses import dataclass
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_learning_lab.task6_retry_layers import ToolCallingFakeModel


DecisionName = Literal["approve", "edit", "reject"]

_cancelled_order_ids: list[str] = []


def reset_cancelled_orders() -> None:
    _cancelled_order_ids.clear()


def cancelled_order_ids() -> list[str]:
    return list(_cancelled_order_ids)


@tool
def cancel_order(order_id: str) -> str:
    """Cancel one order after the caller has obtained required approval."""
    _cancelled_order_ids.append(order_id)
    return f"订单 {order_id} 已取消"


@dataclass
class DecisionDemoResult:
    """Observable state before and after one approval decision."""

    interrupt: dict[str, Any]
    tool_calls_before_resume: list[str]
    tool_calls_after_resume: list[str]
    messages: list[BaseMessage]


def _decision_payload(decision: DecisionName) -> dict[str, Any]:
    if decision == "approve":
        return {"type": "approve"}
    if decision == "edit":
        return {
            "type": "edit",
            "edited_action": {
                "name": "cancel_order",
                "args": {"order_id": "ORDER-2002"},
            },
        }
    return {
        "type": "reject",
        "message": "该订单已经发货，不允许取消。",
    }


def run_decision_demo(decision: DecisionName) -> DecisionDemoResult:
    """Pause one cancellation and resume it with the selected human decision."""
    reset_cancelled_orders()
    final_contents = {
        "approve": "审批通过，订单取消操作已完成。",
        "edit": "审批人修改了订单号，已执行修改后的取消操作。",
        "reject": "审批人拒绝了取消操作，因此订单没有被取消。",
    }
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "cancel_order",
                        "args": {"order_id": "ORDER-1001"},
                        "id": "call_cancel_order",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content=final_contents[decision]),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[cancel_order],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "cancel_order": {
                        "allowed_decisions": ["approve", "edit", "reject"]
                    }
                },
                description_prefix="高风险订单操作等待人工审批",
            )
        ],
        checkpointer=InMemorySaver(),
        name=f"hitl_{decision}_demo",
    )
    config = {"configurable": {"thread_id": f"hitl-{decision}-thread"}}

    paused = agent.invoke(
        {"messages": [{"role": "user", "content": "取消订单 ORDER-1001"}]},
        config=config,
    )
    calls_before_resume = cancelled_order_ids()
    interrupt = paused["__interrupt__"][0].value

    resumed = agent.invoke(
        Command(resume={"decisions": [_decision_payload(decision)]}),
        config=config,
    )
    return DecisionDemoResult(
        interrupt=interrupt,
        tool_calls_before_resume=calls_before_resume,
        tool_calls_after_resume=cancelled_order_ids(),
        messages=resumed["messages"],
    )


def print_message_summary(messages: list[BaseMessage]) -> None:
    for index, message in enumerate(messages, start=1):
        suffix = ""
        if isinstance(message, ToolMessage):
            suffix = f", status={message.status}, tool_call_id={message.tool_call_id}"
        print(f"  [{index}] {message.type}: {message.content!r}{suffix}")


def main() -> None:
    for decision in ("approve", "edit", "reject"):
        result = run_decision_demo(decision)
        request = result.interrupt["action_requests"][0]
        print(f"{decision.upper()}:")
        print("  pending action:", request["name"], request["args"])
        print("  tool calls before resume:", result.tool_calls_before_resume)
        print("  tool calls after resume:", result.tool_calls_after_resume)
        print("  final messages:")
        print_message_summary(result.messages)
        print()


if __name__ == "__main__":
    main()
