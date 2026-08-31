from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver

from langchain_learning_lab.task4_agent_loop import build_agent


def test_create_agent_builds_model_and_tools_nodes() -> None:
    model = ChatDeepSeek(
        model="deepseek-v4-flash",
        api_key="test-key-not-used",
        max_tokens=100,
        extra_body={"thinking": {"type": "disabled"}},
    )

    agent = build_agent(model=model)
    graph = agent.get_graph()

    assert "model" in graph.nodes
    assert "tools" in graph.nodes


def test_create_agent_accepts_short_term_memory_checkpointer() -> None:
    model = ChatDeepSeek(
        model="deepseek-v4-flash",
        api_key="test-key-not-used",
        max_tokens=100,
        extra_body={"thinking": {"type": "disabled"}},
    )
    checkpointer = InMemorySaver()

    agent = build_agent(model=model, checkpointer=checkpointer)

    assert agent.checkpointer is checkpointer
