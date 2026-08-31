"""Task 6: return a validated Pydantic object from create_agent."""

import json
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from langchain_learning_lab.task3_manual_tool_loop import create_model
from langchain_learning_lab.task6_structured_output import SupportTicket, TICKET_TEXT


SYSTEM_PROMPT = (
    "你负责整理客户支持工单。只能提取输入中明确提供的信息，不要虚构。"
    "优先级应根据业务影响和紧急程度判断，最终必须提交结构化工单。"
)


def create_ticket_agent(model: BaseChatModel | None = None):
    """Build an agent that delivers SupportTicket as its final contract."""
    return create_agent(
        model=model or create_model(),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(
            SupportTicket,
            handle_errors=True,
            tool_message_content="Structured support ticket accepted.",
        ),
        name="support_ticket_agent",
    )


def describe_message(index: int, message: BaseMessage) -> None:
    """Display how structured output is represented in Agent message state."""
    print(f"\n[{index}] {message.type.upper()}")
    print("content:", repr(message.content))

    if isinstance(message, AIMessage) and message.tool_calls:
        print("tool_calls:")
        for tool_call in message.tool_calls:
            print("  name:", tool_call["name"])
            print("  args:", tool_call["args"])
            print("  id:", tool_call["id"])

    if isinstance(message, ToolMessage):
        print("tool_call_id:", message.tool_call_id)


def print_agent_result(result: dict[str, Any]) -> None:
    """Separate the diagnostic message timeline from the business result."""
    print("Final Agent State keys:", sorted(result))
    print("\nMessage timeline:")
    for index, message in enumerate(result["messages"], start=1):
        describe_message(index, message)

    structured: SupportTicket = result["structured_response"]
    print("\nstructured_response type:", type(structured).__name__)
    print(
        "structured_response:\n",
        json.dumps(structured.model_dump(), ensure_ascii=False, indent=2),
        sep="",
    )


def main() -> None:
    agent = create_ticket_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": TICKET_TEXT}]},
        config={"recursion_limit": 6},
    )
    print_agent_result(result)


if __name__ == "__main__":
    main()
