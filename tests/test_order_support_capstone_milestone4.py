from typing import Any, Literal

from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_learning_lab.task6_retry_layers import ToolCallingFakeModel
from order_support_capstone.agent import build_order_support_agent
from order_support_capstone.domain import OrderStatus, ServiceResolution
from order_support_capstone.repository import OrderRepository


DecisionName = Literal["approve", "edit", "reject"]


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


def resolution_call(
    *,
    order_id: str,
    action_taken: Literal["cancelled", "rejected"],
    answer: str,
) -> AIMessage:
    return tool_call(
        "ServiceResolution",
        {
            "answer": answer,
            "order_id": order_id,
            "action_taken": action_taken,
            "needs_human_approval": False,
            "sources": [],
        },
        "call_resolution",
    )


def build_cancellation_case(
    *,
    requested_order_id: str,
    final_order_id: str,
    final_action: Literal["cancelled", "rejected"],
):
    model = ToolCallingFakeModel(
        responses=[
            tool_call(
                "cancel_order",
                {
                    "order_id": requested_order_id,
                    "reason": "用户不再需要",
                },
                "call_cancel_order",
            ),
            resolution_call(
                order_id=final_order_id,
                action_taken=final_action,
                answer=f"审批结果：{final_action}",
            ),
        ]
    )
    repository = OrderRepository.with_sample_orders()
    agent = build_order_support_agent(
        model=model,
        repository=repository,
        embeddings=DeterministicFakeEmbedding(size=64),
        checkpointer=InMemorySaver(),
    )
    config = {
        "configurable": {
            "thread_id": f"milestone-4-{requested_order_id}-{final_action}"
        }
    }
    return agent, repository, config


def start_review(agent, repository: OrderRepository, config: dict) -> dict:
    paused = agent.invoke(
        {"messages": [{"role": "user", "content": "请取消订单"}]},
        config=config,
    )

    assert "__interrupt__" in paused
    assert repository.cancel_count == 0
    return paused["__interrupt__"][0].value


def resume_review(agent, config: dict, decision: dict[str, Any]) -> dict:
    return agent.invoke(
        Command(resume={"decisions": [decision]}),
        config=config,
    )


def test_agent_graph_contains_limits_and_hitl_nodes() -> None:
    agent, _, _ = build_cancellation_case(
        requested_order_id="ORDER-1001",
        final_order_id="ORDER-1001",
        final_action="cancelled",
    )

    nodes = set(agent.get_graph().nodes)

    assert "ModelCallLimitMiddleware.before_model" in nodes
    assert "ToolCallLimitMiddleware[search_order_policy].after_model" in nodes
    assert "HumanInTheLoopMiddleware.after_model" in nodes


def test_approve_executes_original_cancellation_after_resume() -> None:
    agent, repository, config = build_cancellation_case(
        requested_order_id="ORDER-1001",
        final_order_id="ORDER-1001",
        final_action="cancelled",
    )

    interrupt = start_review(agent, repository, config)
    assert interrupt["action_requests"][0]["args"]["order_id"] == "ORDER-1001"

    result = resume_review(agent, config, {"type": "approve"})

    assert repository.query("ORDER-1001").status == OrderStatus.CANCELLED
    assert repository.cancel_count == 1
    assert isinstance(result["structured_response"], ServiceResolution)
    assert result["structured_response"].action_taken == "cancelled"


def test_edit_executes_modified_order_and_not_original_order() -> None:
    agent, repository, config = build_cancellation_case(
        requested_order_id="ORDER-1002",
        final_order_id="ORDER-1001",
        final_action="cancelled",
    )
    start_review(agent, repository, config)

    result = resume_review(
        agent,
        config,
        {
            "type": "edit",
            "edited_action": {
                "name": "cancel_order",
                "args": {
                    "order_id": "ORDER-1001",
                    "reason": "审批人改为取消正确订单",
                },
            },
        },
    )

    assert repository.query("ORDER-1001").status == OrderStatus.CANCELLED
    assert repository.query("ORDER-1002").status == OrderStatus.SHIPPED
    assert repository.cancel_count == 1
    assert result["structured_response"].order_id == "ORDER-1001"


def test_reject_returns_error_feedback_without_mutating_repository() -> None:
    agent, repository, config = build_cancellation_case(
        requested_order_id="ORDER-1001",
        final_order_id="ORDER-1001",
        final_action="rejected",
    )
    start_review(agent, repository, config)

    result = resume_review(
        agent,
        config,
        {
            "type": "reject",
            "message": "订单仍然需要，不允许取消。",
        },
    )

    assert repository.query("ORDER-1001").status == OrderStatus.PENDING
    assert repository.cancel_count == 0
    assert result["structured_response"].action_taken == "rejected"
    rejection = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.name == "cancel_order"
    )
    assert rejection.status == "error"
    assert "不允许取消" in str(rejection.content)
