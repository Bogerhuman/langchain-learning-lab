"""Task 4: let create_agent manage the model/tool loop automatically."""

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from langchain_learning_lab.task3_manual_tool_loop import TOOLS, create_model


SYSTEM_PROMPT = (
    "You are an order assistant. Use the provided tools for every monetary "
    "calculation. Never invent a computed amount. Explain verified results in Chinese."
)


def build_agent(model: BaseChatModel | None = None, checkpointer=None):
    """Build the LangGraph-backed agent with the same tools used in Task 3."""
    return create_agent(
        model=model or create_model(),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        name="order_assistant",
        checkpointer=checkpointer,
    )


def describe_message(index: int, message: BaseMessage) -> None:
    """Print the important state carried by each message type."""
    print(f"\n[{index}] {message.type.upper()}")

    if isinstance(message, HumanMessage):
        print("content:", message.content)
        return

    if isinstance(message, AIMessage):
        print("content:", repr(message.content))
        if message.tool_calls:
            print("tool_calls:")
            for tool_call in message.tool_calls:
                print("  name:", tool_call["name"])
                print("  args:", tool_call["args"])
                print("  id:", tool_call["id"])
        return

    if isinstance(message, ToolMessage):
        print("name:", message.name)
        print("tool_call_id:", message.tool_call_id)
        print("content:", message.content)
        return

    print("content:", message.content)


def main() -> None:
    agent = build_agent()
    user_question = (
        "请分别计算两个订单："
        "订单 A 单价 79.9 元，买 3 件，打 85 折；"
        "订单 B 单价 120 元，买 2 件，满 200 减 30。"
    )

    # Only the new HumanMessage is supplied. The agent manages every later step.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_question}]}
    )

    print("Final state message timeline:")
    for index, message in enumerate(result["messages"], start=1):
        describe_message(index, message)


if __name__ == "__main__":
    main()
