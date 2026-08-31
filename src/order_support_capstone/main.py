"""Manual entry point for the finished capstone application."""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from langchain_learning_lab.task3_manual_tool_loop import create_model
from langchain_learning_lab.task5_retrieval import create_embeddings
from order_support_capstone import OrderRepository
from order_support_capstone.agent import build_order_support_agent


def main() -> None:
    repository = OrderRepository.with_sample_orders()
    checkpointer = InMemorySaver()

    agent = build_order_support_agent(
        model=create_model(),
        repository=repository,
        embeddings=create_embeddings(),
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": "order-support-demo"
        }
    }

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "查询 ORDER-1001 的状态，并告诉我它是否可以取消",
                }
            ]
        },
        config=config,
    )
    resolution = result["structured_response"]

    print(
        resolution.model_dump_json(
            indent=2,
        )
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请取消 ORDER-1001，原因是我不再需要了",
                }
            ]
        },
        config=config,
    )

    interrupts = result.get("__interrupt__", [])

    if not interrupts:
        raise RuntimeError("取消操作没有触发人工审批")

    interrupt = interrupts[0].value

    for action in interrupt["action_requests"]:
        print("待审批工具：", action["name"])
        print("参数：", action["args"])
        print("说明：", action["description"])

    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                ]
            }
        ),
        config=config,
    )

    resolution = result.get("structured_response")

    if resolution is None:
        raise RuntimeError("审批恢复后未生成结构化结果")

    print(resolution.model_dump_json(indent=2))
    print(repository.query("ORDER-1001").model_dump_json(indent=2))
    print("成功取消次数：", repository.cancel_count)


if __name__ == "__main__":
    main()
