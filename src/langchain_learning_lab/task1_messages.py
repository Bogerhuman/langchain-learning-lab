"""A provider-free example of LangChain's message abstraction."""

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def build_messages(topic: str) -> list[SystemMessage | HumanMessage | AIMessage]:
    """Build the model-independent messages that would be sent to an LLM."""
    return [
        SystemMessage(content="You are a experienced Python application tutor."),
        HumanMessage(content=f"Explain {topic} in one sentence."),
        AIMessage(
            content=(
                "This is a mock response, not an actual model output. "
                "LangChain provides high-level AI application abstractions, "
                "while LangGraph provides the lower-level stateful workflow runtime."
            )
        )
    ]


def main() -> None:
    for message in build_messages("the difference between LangChain and an LangGraph"):
        print(f"{message.type}: {message.content}")


if __name__ == "__main__":
    main()
