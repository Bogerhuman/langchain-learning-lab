from langchain_core.messages import ToolMessage

import pytest

from langchain_learning_lab.task3_manual_tool_loop import (
    TOOLS_BY_NAME,
    calculate_fixed_reduction_total,
    calculate_order_total,
    execute_tool_call,
)


def test_calculate_order_total_returns_verified_amounts() -> None:
    result = calculate_order_total.invoke(
        {
            "unit_price_yuan": 79.9,
            "quantity": 3,
            "discount_percent": 15,
        }
    )

    assert result["subtotal_yuan"] == 239.7
    assert result["total_yuan"] == 203.75


def test_tool_call_invocation_preserves_call_id() -> None:
    result = calculate_order_total.invoke(
        {
            "name": "calculate_order_total",
            "args": {
                "unit_price_yuan": 10,
                "quantity": 2,
                "discount_percent": 0,
            },
            "id": "call_test_123",
            "type": "tool_call",
        }
    )

    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call_test_123"
    assert "20" in str(result.content)


def test_fixed_reduction_returns_verified_amounts() -> None:
    result = calculate_fixed_reduction_total.invoke(
        {
            "unit_price_yuan": 79.9,
            "quantity": 3,
            "minimum_spend_yuan": 200,
            "reduction_yuan": 30,
        }
    )

    assert result["subtotal_yuan"] == 239.7
    assert result["total_yuan"] == 209.7
    assert result["reduction_applied"] is True


def test_fixed_reduction_is_not_applied_below_threshold() -> None:
    result = calculate_fixed_reduction_total.invoke(
        {
            "unit_price_yuan": 79.9,
            "quantity": 2,
            "minimum_spend_yuan": 200,
            "reduction_yuan": 30,
        }
    )

    assert result["subtotal_yuan"] == 159.8
    assert result["total_yuan"] == 159.8
    assert result["reduction_applied"] is False


def test_tool_registry_contains_both_discount_strategies() -> None:
    assert set(TOOLS_BY_NAME) == {
        "calculate_order_total",
        "calculate_fixed_reduction_total",
    }


def test_dispatcher_rejects_unknown_tool() -> None:
    with pytest.raises(ValueError, match="Unknown or disallowed tool"):
        execute_tool_call(
            {
                "name": "run_arbitrary_code",
                "args": {},
                "id": "call_untrusted",
                "type": "tool_call",
            }
        )


def test_each_tool_call_produces_a_matching_tool_message() -> None:
    tool_calls = [
        {
            "name": "calculate_order_total",
            "args": {
                "unit_price_yuan": 79.9,
                "quantity": 3,
                "discount_percent": 15,
            },
            "id": "call_percentage",
            "type": "tool_call",
        },
        {
            "name": "calculate_fixed_reduction_total",
            "args": {
                "unit_price_yuan": 120,
                "quantity": 2,
                "minimum_spend_yuan": 200,
                "reduction_yuan": 30,
            },
            "id": "call_fixed_reduction",
            "type": "tool_call",
        },
    ]

    results = [execute_tool_call(tool_call) for tool_call in tool_calls]

    assert [result.tool_call_id for result in results] == [
        "call_percentage",
        "call_fixed_reduction",
    ]
    assert "203.75" in str(results[0].content)
    assert "210.0" in str(results[1].content)
