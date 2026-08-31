"""Task 6: compare free-form model text with validated structured output."""

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from langchain_learning_lab.task3_manual_tool_loop import create_model


class SupportTicket(BaseModel):
    """Validated information extracted from one customer support request."""

    model_config = ConfigDict(extra="forbid")

    category: Literal["account", "billing", "database", "other"] = Field(
        description="The primary category of the reported problem"
    )
    priority: Literal["low", "medium", "high", "critical"] = Field(
        description="Business impact and urgency"
    )
    summary: str = Field(
        min_length=10,
        max_length=160,
        description="A concise Chinese summary based only on the request",
    )
    affected_services: list[str] = Field(
        min_length=1,
        description="Services explicitly described as affected",
    )
    requires_human: bool = Field(
        description="Whether a human operator must handle or approve the case"
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in this classification from 0 to 1",
    )


TICKET_TEXT = (
    "生产环境的支付 API 已连续一小时返回 500。排查发现数据库连接池耗尽，"
    "目前客户无法完成付款，需要值班工程师立即介入。"
)


def build_ticket_messages(ticket_text: str) -> list[BaseMessage]:
    """Build identical input messages for plain and structured model calls."""
    return [
        SystemMessage(
            content=(
                "你负责整理客户支持工单。只能提取输入中明确提供的信息，不要虚构。"
                "优先级应根据业务影响和紧急程度判断。"
            )
        ),
        HumanMessage(content=ticket_text),
    ]


def invoke_structured(
    model: ChatDeepSeek,
    messages: list[BaseMessage],
) -> dict[str, Any]:
    """Return raw response, parsed Pydantic object, and any parsing error."""
    structured_model = model.with_structured_output(
        SupportTicket,
        method="function_calling",
        include_raw=True,
    )
    return structured_model.invoke(messages)


def print_structured_result(result: dict[str, Any]) -> None:
    """Display the boundary between provider output and validated business data."""
    raw: AIMessage = result["raw"]
    parsed: SupportTicket | None = result["parsed"]
    parsing_error = result["parsing_error"]

    print("\nStructured result container keys:", list(result))
    print("Raw response type:", raw.type)
    print("Raw content:", repr(raw.content))
    print("Raw tool calls:", raw.tool_calls)
    print("Raw token usage:", raw.usage_metadata)
    print("Parsing error:", repr(parsing_error))

    if parsed is None:
        print("Parsed object: None")
        return

    print("Parsed object type:", type(parsed).__name__)
    print(
        "Parsed object:\n",
        json.dumps(parsed.model_dump(), ensure_ascii=False, indent=2),
        sep="",
    )
    print("Business field access: priority =", parsed.priority)


def demonstrate_local_validation() -> None:
    """Show that business-invalid data is rejected even when Python types match."""
    try:
        SupportTicket(
            category="database",
            priority="critical",
            summary="支付服务因数据库连接池耗尽而不可用",
            affected_services=["payment-api"],
            requires_human=True,
            confidence=1.5,
        )
    except ValidationError as error:
        first_error = error.errors()[0]
        print("\nLocal Pydantic validation rejected confidence=1.5")
        print("Error type:", first_error["type"])
        print("Field path:", first_error["loc"])
        print("Message:", first_error["msg"])


def main() -> None:
    model = create_model()
    messages = build_ticket_messages(TICKET_TEXT)

    plain_response = model.invoke(messages)
    print("Plain response type:", type(plain_response).__name__)
    print("Plain response content:\n", plain_response.content, sep="")

    structured_result = invoke_structured(model, messages)
    print_structured_result(structured_result)
    demonstrate_local_validation()


if __name__ == "__main__":
    main()
