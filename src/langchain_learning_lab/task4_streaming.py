"""Task 4: stream LangGraph agent progress one node update at a time."""

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain_learning_lab.task4_agent_loop import build_agent


def describe_update(node_name: str, messages: list[BaseMessage]) -> list[str]:
    """Convert one node update into concise, testable display lines."""
    lines = [f"Node: {node_name}"]
    for message in messages:
        lines.append(f"  Message type: {message.type}")

        if isinstance(message, AIMessage) and message.tool_calls:
            for tool_call in message.tool_calls:
                lines.append(
                    f"  Tool call: {tool_call['name']} "
                    f"args={tool_call['args']} id={tool_call['id']}"
                )
        elif isinstance(message, ToolMessage):
            lines.append(
                f"  Tool result: name={message.name} "
                f"id={message.tool_call_id} content={message.content}"
            )
        elif message.content:
            lines.append(f"  Content: {message.content}")
    return lines


def main() -> None:
    agent = build_agent(checkpointer=InMemorySaver())
    config = {
        "configurable": {"thread_id": "task4-stream-demo"},
        # Stop runaway model -> tools -> model loops instead of running forever.
        "recursion_limit": 10,
    }
    user_input = {
        "messages": [
            {
                "role": "user",
                "content": "商品单价79.9元，买3件，打85折，最终多少钱？",
            }
        ]
    }

    node_sequence: list[str] = []
    for update in agent.stream(
        user_input,
        config=config,
        stream_mode="updates",
    ):
        for node_name, node_update in update.items():
            node_sequence.append(node_name)
            for line in describe_update(node_name, node_update.get("messages", [])):
                print(line)

    snapshot = agent.get_state(config)
    print("\nObserved node sequence:", " -> ".join(node_sequence))
    print("Saved message count:", len(snapshot.values.get("messages", [])))
    print("Next nodes after completion:", snapshot.next)


if __name__ == "__main__":
    main()
