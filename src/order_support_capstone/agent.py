"""Composition root for the completed order-support Agent."""

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware, InterruptOnConfig,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver

from order_support_capstone.domain import ServiceResolution
from order_support_capstone.order_tools import build_order_tools
from order_support_capstone.repository import OrderRepository
from order_support_capstone.retrieval import (
    build_policy_retriever,
    build_policy_search_tool,
    load_policy_document,
    split_policy_documents,
)

SYSTEM_PROMPT = """You are an order-support assistant.
Use tools for order facts and company policy instead of inventing information.
Never claim a cancellation succeeded until the cancellation tool reports success.
"""


def build_order_support_agent(
        *,
        model: BaseChatModel,
        repository: OrderRepository,
        embeddings: Embeddings,
        checkpointer: BaseCheckpointSaver,
):
    """Assemble RAG, tools, middleware, memory, HITL, and output schema."""

    order_tools = build_order_tools(repository)
    documents = load_policy_document()
    chunks = split_policy_documents(documents)
    retriever = build_policy_retriever(chunks, embeddings)
    policy_tool = build_policy_search_tool(retriever)

    tools = [
        *order_tools,
        policy_tool,
    ]

    middleware = [ModelRetryMiddleware(
        max_retries=2,
        retry_on=(TimeoutError, ConnectionError),
        on_failure="error",
        initial_delay=0,
        backoff_factor=0,
        jitter=False,
    ),

        ToolRetryMiddleware(
            max_retries=2,
            tools=["query_order", "search_order_policy"],
            retry_on=(TimeoutError, ConnectionError),
            on_failure="error",
            initial_delay=0,
            backoff_factor=0,
            jitter=False,
        ),

        ModelCallLimitMiddleware(
            run_limit=6,
            exit_behavior="end",
        ),

        ToolCallLimitMiddleware(
            tool_name="search_order_policy",
            run_limit=3,
            exit_behavior="continue",
        ),

        HumanInTheLoopMiddleware(
            interrupt_on={
                "cancel_order": InterruptOnConfig(
                    allowed_decisions=[
                        "approve",
                        "edit",
                        "reject",
                    ]
                )
            },
            description_prefix="订单取消操作等待人工审批",
        )]

    return create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(
            ServiceResolution,
            handle_errors=True,
        ),
        checkpointer=checkpointer,
        name="order_support_agent",
        middleware=middleware,
    )
