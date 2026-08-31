"""Task 4: keep short-term agent state isolated by thread_id."""

from langgraph.checkpoint.memory import InMemorySaver

from langchain_learning_lab.task4_agent_loop import build_agent


THREAD_A = {"configurable": {"thread_id": "customer-session-a"}}
THREAD_B = {"configurable": {"thread_id": "customer-session-b"}}


def invoke_turn(agent, content: str, config: dict) -> str:
    """Send only the new user message; prior state comes from the checkpointer."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
        config=config,
    )
    return str(result["messages"][-1].content)


def print_thread_state(agent, label: str, config: dict) -> None:
    """Inspect the messages currently checkpointed for one thread."""
    snapshot = agent.get_state(config)
    messages = snapshot.values.get("messages", [])
    print(f"{label} saved message count: {len(messages)}")
    print(f"{label} message types: {[message.type for message in messages]}")


def main() -> None:
    checkpointer = InMemorySaver()
    agent = build_agent(checkpointer=checkpointer)

    first_answer = invoke_turn(
        agent,
        "请记住：客户叫小林，会员折扣是85折。这里只需确认记住，不要计算订单。",
        THREAD_A,
    )
    print("Thread A, turn 1:", first_answer)

    recalled_answer = invoke_turn(
        agent,
        "刚才的客户叫什么？会员折扣是多少？",
        THREAD_A,
    )
    print("Thread A, turn 2:", recalled_answer)

    isolated_answer = invoke_turn(
        agent,
        "刚才的客户叫什么？会员折扣是多少？如果没有上下文，请明确说不知道。",
        THREAD_B,
    )
    print("Thread B, turn 1:", isolated_answer)

    print_thread_state(agent, "Thread A", THREAD_A)
    print_thread_state(agent, "Thread B", THREAD_B)


if __name__ == "__main__":
    main()
