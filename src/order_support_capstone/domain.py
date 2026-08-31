"""Domain types and structured delivery contracts for the capstone project."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OrderStatus(StrEnum):
    PENDING = "pending"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"


class Order(BaseModel):
    """Mutable teaching model stored by the in-memory repository."""

    model_config = ConfigDict(validate_assignment=True)

    order_id: str = Field(pattern=r"^ORDER-\d{4}$")
    status: OrderStatus


class OrderQueryResult(BaseModel):
    """Stable business response returned by the query tool."""

    success: bool
    order_id: str
    status: OrderStatus | None = None
    error_code: Literal["ORDER_NOT_FOUND"] | None = None
    message: str


class CancelOrderResult(BaseModel):
    """Stable business response returned by the cancellation tool."""

    success: bool
    order_id: str
    previous_status: OrderStatus | None = None
    current_status: OrderStatus | None = None
    error_code: Literal[
        "ORDER_NOT_FOUND",
        "REASON_REQUIRED",
        "ORDER_ALREADY_CANCELLED",
        "ORDER_ALREADY_SHIPPED",
    ] | None = None
    message: str


class ServiceResolution(BaseModel):
    """Final validated contract delivered by the completed Agent."""

    answer: str = Field(min_length=1)
    order_id: str | None = Field(default=None, pattern=r"^ORDER-\d{4}$")
    action_taken: Literal["queried", "cancelled", "rejected", "none"]
    needs_human_approval: bool
    sources: list[str] = Field(default_factory=list)
