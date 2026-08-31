from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import Runnable
from pydantic import Field
from langgraph.checkpoint.memory import InMemorySaver

from langchain_learning_lab.task6_retry_layers import ToolCallingFakeModel
from order_support_capstone.agent import build_order_support_agent
from order_support_capstone.domain import ServiceResolution
from order_support_capstone.repository import OrderRepository


class RecordingToolCallingFakeModel(ToolCallingFakeModel):
    """Record every tool schema exposed by create_agent."""

    bound_tool_names: list[str] = Field(default_factory=list)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any]],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable:
        names: list[str] = []
        for item in tools:
            if hasattr(item, "name"):
                names.append(item.name)
            elif isinstance(item, dict):
                names.append(item.get("name") or item.get("function", {}).get("name"))
            elif isinstance(item, type):
                names.append(item.__name__)
        self.bound_tool_names = names
        return self


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def build_test_agent(model: RecordingToolCallingFakeModel):
    repository = OrderRepository.with_sample_orders()
    agent = build_order_support_agent(
        model=model,
        repository=repository,
        embeddings=DeterministicFakeEmbedding(size=64),
        checkpointer=InMemorySaver(),
    )
    return agent, repository


def test_agent_graph_contains_model_and_tools_nodes() -> None:
    model = RecordingToolCallingFakeModel(
        responses=[
            tool_call(
                "ServiceResolution",
                {
                    "answer": "无需执行工具。",
                    "order_id": None,
                    "action_taken": "none",
                    "needs_human_approval": False,
                    "sources": [],
                },
                "call_resolution",
            )
        ]
    )
    agent, _ = build_test_agent(model)

    graph = agent.get_graph()

    assert "model" in graph.nodes
    assert "tools" in graph.nodes


def test_query_order_then_deliver_structured_resolution() -> None:
    model = RecordingToolCallingFakeModel(
        responses=[
            tool_call(
                "query_order",
                {"order_id": "ORDER-1001"},
                "call_query_order",
            ),
            tool_call(
                "ServiceResolution",
                {
                    "answer": "ORDER-1001 当前状态为 pending。",
                    "order_id": "ORDER-1001",
                    "action_taken": "queried",
                    "needs_human_approval": False,
                    "sources": [],
                },
                "call_resolution",
            ),
        ]
    )
    agent, repository = build_test_agent(model)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "查询 ORDER-1001 状态"}]},
        config={"configurable": {"thread_id": "milestone-3-query"}},
    )

    assert set(model.bound_tool_names) == {
        "query_order",
        "cancel_order",
        "search_order_policy",
        "ServiceResolution",
    }
    assert isinstance(result["structured_response"], ServiceResolution)
    assert result["structured_response"].order_id == "ORDER-1001"
    assert result["structured_response"].action_taken == "queried"
    assert repository.cancel_count == 0

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    assert tool_messages[0].name == "query_order"
    assert "pending" in str(tool_messages[0].content)
    assert tool_messages[1].name == "ServiceResolution"
