import pytest

from order_support_capstone.domain import OrderStatus
from order_support_capstone.order_tools import build_order_tools
from order_support_capstone.repository import OrderRepository


@pytest.fixture
def repository() -> OrderRepository:
    return OrderRepository.with_sample_orders()


def test_query_existing_order(repository: OrderRepository) -> None:
    result = repository.query("ORDER-1001")

    assert result.success is True
    assert result.status == OrderStatus.PENDING
    assert result.error_code is None


def test_query_unknown_order_returns_stable_error(repository: OrderRepository) -> None:
    result = repository.query("ORDER-9999")

    assert result.success is False
    assert result.status is None
    assert result.error_code == "ORDER_NOT_FOUND"


def test_pending_order_can_be_cancelled(repository: OrderRepository) -> None:
    result = repository.cancel("ORDER-1001", "用户不再需要")

    assert result.success is True
    assert result.previous_status == OrderStatus.PENDING
    assert result.current_status == OrderStatus.CANCELLED
    assert repository.cancel_count == 1


def test_shipped_order_cannot_be_cancelled(repository: OrderRepository) -> None:
    result = repository.cancel("ORDER-1002", "地址填写错误")

    assert result.success is False
    assert result.error_code == "ORDER_ALREADY_SHIPPED"
    assert repository.query("ORDER-1002").status == OrderStatus.SHIPPED
    assert repository.cancel_count == 0


def test_cancelled_order_is_idempotent(repository: OrderRepository) -> None:
    first = repository.cancel("ORDER-1003", "重复申请")

    assert first.success is False
    assert first.error_code == "ORDER_ALREADY_CANCELLED"
    assert repository.cancel_count == 0


@pytest.mark.parametrize("reason", ["", " ", "\n\t"])
def test_blank_reason_is_rejected(
    repository: OrderRepository, reason: str
) -> None:
    result = repository.cancel("ORDER-1001", reason)

    assert result.success is False
    assert result.error_code == "REASON_REQUIRED"
    assert repository.query("ORDER-1001").status == OrderStatus.PENDING
    assert repository.cancel_count == 0


def test_successful_cancellation_changes_state_only_once(
    repository: OrderRepository,
) -> None:
    first = repository.cancel("ORDER-1001", "用户不再需要")
    second = repository.cancel("ORDER-1001", "重复请求")

    assert first.success is True
    assert second.success is False
    assert second.error_code == "ORDER_ALREADY_CANCELLED"
    assert repository.query("ORDER-1001").status == OrderStatus.CANCELLED
    assert repository.cancel_count == 1


def test_tools_have_schema_and_delegate_to_repository(
    repository: OrderRepository,
) -> None:
    tools = {item.name: item for item in build_order_tools(repository)}

    assert set(tools) == {"query_order", "cancel_order"}
    assert tools["query_order"].description
    assert tools["cancel_order"].description
    assert set(tools["query_order"].args) == {"order_id"}
    assert set(tools["cancel_order"].args) == {"order_id", "reason"}

    query_result = tools["query_order"].invoke({"order_id": "ORDER-1001"})
    cancel_result = tools["cancel_order"].invoke(
        {"order_id": "ORDER-1001", "reason": "用户不再需要"}
    )

    assert query_result["status"] == "pending"
    assert cancel_result["success"] is True
    assert cancel_result["current_status"] == "cancelled"
