import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from langchain_learning_lab.task6_structured_output import (
    SupportTicket,
    build_ticket_messages,
)


VALID_TICKET = {
    "category": "database",
    "priority": "critical",
    "summary": "支付服务因数据库连接池耗尽而不可用",
    "affected_services": ["payment-api"],
    "requires_human": True,
    "confidence": 0.95,
}


def test_support_ticket_accepts_valid_business_data() -> None:
    ticket = SupportTicket.model_validate(VALID_TICKET)

    assert ticket.priority == "critical"
    assert ticket.requires_human is True
    assert ticket.confidence == 0.95


@pytest.mark.parametrize("priority", ["urgent", "p0", "最高"])
def test_support_ticket_rejects_priority_outside_literal(priority: str) -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        SupportTicket.model_validate({**VALID_TICKET, "priority": priority})


@pytest.mark.parametrize("confidence", [-0.01, 1.01, 1.5])
def test_support_ticket_rejects_confidence_outside_range(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        SupportTicket.model_validate(
            {**VALID_TICKET, "confidence": confidence}
        )


def test_support_ticket_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SupportTicket.model_validate({**VALID_TICKET, "invented": "value"})


def test_schema_exposes_enum_and_numeric_constraints_to_model() -> None:
    schema = SupportTicket.model_json_schema()

    assert schema["properties"]["priority"]["enum"] == [
        "low",
        "medium",
        "high",
        "critical",
    ]
    assert schema["properties"]["confidence"]["minimum"] == 0
    assert schema["properties"]["confidence"]["maximum"] == 1
    assert set(schema["required"]) == set(VALID_TICKET)


def test_build_ticket_messages_keeps_roles_and_input() -> None:
    messages = build_ticket_messages("数据库无法连接")

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "数据库无法连接"
