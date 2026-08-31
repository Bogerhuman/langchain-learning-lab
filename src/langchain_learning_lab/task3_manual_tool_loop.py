"""Task 3: inspect, route, and execute multiple tools manually."""

import json
from decimal import Decimal, ROUND_HALF_UP

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_deepseek import ChatDeepSeek

from langchain_learning_lab.credentials import ensure_project_credential
from langchain_learning_lab.task2_deepseek import MODEL_NAME


@tool
def calculate_order_total(
    unit_price_yuan: float,
    quantity: int,
    discount_percent: int = 0,
) -> dict[str, float | int]:
    """Calculate a total after a percentage discount in Chinese yuan.

    Use for percentage discounts such as 85折 or 15% off.
    Do not use for fixed reductions such as 满200减30 or 立减30元.

    Args:
        unit_price_yuan: Price of one item in Chinese yuan.
        quantity: Number of items purchased; must be positive.
        discount_percent: Integer discount from 0 to 100, such as 15 for 15% off.
    """
    if unit_price_yuan < 0:
        raise ValueError("unit_price_yuan must not be negative")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if not 0 <= discount_percent <= 100:
        raise ValueError("discount_percent must be between 0 and 100")

    cents = Decimal("0.01")
    unit_price = Decimal(str(unit_price_yuan))
    subtotal = (unit_price * quantity).quantize(cents, rounding=ROUND_HALF_UP)
    discount_multiplier = Decimal(100 - discount_percent) / Decimal(100)
    total = (subtotal * discount_multiplier).quantize(
        cents,
        rounding=ROUND_HALF_UP,
    )
    return {
        "unit_price_yuan": unit_price_yuan,
        "quantity": quantity,
        "discount_percent": discount_percent,
        "subtotal_yuan": float(subtotal),
        "total_yuan": float(total),
    }


@tool
def calculate_fixed_reduction_total(
    unit_price_yuan: float,
    quantity: int,
    minimum_spend_yuan: float,
    reduction_yuan: float,
) -> dict[str, float | int]:
    """Calculate a total after a fixed-amount reduction in Chinese yuan.

    Use for fixed reductions such as 满200减30 or 立减30元.
    Do not use for percentage discounts such as 85折 or 15% off.

    Args:
        unit_price_yuan: Price of one item in Chinese yuan.
        quantity: Number of items purchased; must be positive.
        minimum_spend_yuan: Required subtotal for the reduction to apply.
        reduction_yuan: Fixed amount deducted from the subtotal.
    """
    if unit_price_yuan < 0:
        raise ValueError("unit_price_yuan must not be negative")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if minimum_spend_yuan < 0:
        raise ValueError("minimum_spend_yuan must not be negative")
    if reduction_yuan < 0:
        raise ValueError("reduction_yuan must not be negative")

    cents = Decimal("0.01")
    unit_price = Decimal(str(unit_price_yuan))
    minimum_spend = Decimal(str(minimum_spend_yuan))
    reduction = Decimal(str(reduction_yuan))
    subtotal = (unit_price * quantity).quantize(cents, rounding=ROUND_HALF_UP)
    applied_reduction = reduction if subtotal >= minimum_spend else Decimal("0")
    total = max(subtotal - applied_reduction, Decimal("0")).quantize(
        cents,
        rounding=ROUND_HALF_UP,
    )
    return {
        "unit_price_yuan": unit_price_yuan,
        "quantity": quantity,
        "minimum_spend_yuan": minimum_spend_yuan,
        "reduction_yuan": reduction_yuan,
        "reduction_applied": subtotal >= minimum_spend,
        "applied_reduction_yuan": float(applied_reduction),
        "subtotal_yuan": float(subtotal),
        "total_yuan": float(total),
    }


TOOLS = [calculate_order_total, calculate_fixed_reduction_total]
TOOLS_BY_NAME = {registered_tool.name: registered_tool for registered_tool in TOOLS}


def create_model() -> ChatDeepSeek:
    """Create a non-thinking model so the tool-calling flow stays concise."""
    ensure_project_credential("DEEPSEEK_API_KEY")

    return ChatDeepSeek(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=500,
        timeout=30,
        max_retries=2,
        extra_body={"thinking": {"type": "disabled"}},
    )


def execute_tool_call(tool_call: dict):
    """Validate the model-selected name against the allowlist, then execute it."""
    tool_name = tool_call["name"]
    if tool_name not in TOOLS_BY_NAME:
        raise ValueError(f"Unknown or disallowed tool: {tool_name}")
    return TOOLS_BY_NAME[tool_name].invoke(tool_call)


def run_order_assistant(user_question: str) -> None:
    """Run one manual model -> multiple tools -> model exchange."""
    model_with_tools = create_model().bind_tools(TOOLS)

    messages = [
        SystemMessage(
            content=(
                "You are an order assistant. Always use the provided calculator "
                "for arithmetic, then explain the verified result in Chinese."
            )
        ),
        HumanMessage(content=user_question),
    ]

    # Step 1: the model requests a tool call; it does not execute Python itself.
    tool_request = model_with_tools.invoke(messages)
    messages.append(tool_request)

    print("Step 1 - model response content:", repr(tool_request.content))
    print(f"Step 1 - requested {len(tool_request.tool_calls)} tool call(s):")
    for index, tool_call in enumerate(tool_request.tool_calls, start=1):
        print(f"  [{index}] name:", tool_call["name"])
        print(f"  [{index}] args:", tool_call["args"])
        print(f"  [{index}] id:", tool_call["id"])

    if not tool_request.tool_calls:
        raise RuntimeError("The model did not request a tool call")

    # Step 2: application code selects and executes each requested tool.
    for index, tool_call in enumerate(tool_request.tool_calls, start=1):
        tool_result = execute_tool_call(tool_call)
        messages.append(tool_result)
        print(f"Step 2 - result [{index}] type:", tool_result.type)
        print(f"Step 2 - result [{index}] content:", tool_result.content)
        print(f"Step 2 - result [{index}] call id:", tool_result.tool_call_id)

    # Step 3: the model reads the ToolMessage and writes the user-facing answer.
    final_response = model_with_tools.invoke(messages)
    print("Step 3 - final answer:", final_response.content)


def main() -> None:
    for registered_tool in TOOLS:
        print(f"\nTool: {registered_tool.name}")
        print("Description:", registered_tool.description)
        print(
            "Input schema:",
            json.dumps(
                registered_tool.args_schema.model_json_schema(),
                ensure_ascii=False,
                indent=2,
            ),
        )

    question = (
        "请分别计算两个互不相关的订单，并分别给出最终金额："
        "订单 A：商品单价 79.9 元，买 3 件，打 85 折；"
        "订单 B：商品单价 120 元，买 2 件，参加满 200 减 30。"
    )
    print("\n" + "=" * 72)
    print("User question:", question)
    run_order_assistant(question)


if __name__ == "__main__":
    main()
