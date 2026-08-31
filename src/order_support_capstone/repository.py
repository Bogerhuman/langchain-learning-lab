"""In-memory order repository containing the capstone's business boundary."""

from collections.abc import Iterable

from order_support_capstone.domain import (
    CancelOrderResult,
    Order,
    OrderQueryResult,
    OrderStatus,
)


class OrderRepository:
    """Store orders and enforce cancellation invariants.

    The LLM must never be trusted to enforce these rules. This class is the
    authoritative server-side boundary even when it is called without an Agent.
    """

    def __init__(self, orders: Iterable[Order] | None = None) -> None:
        self._orders = {order.order_id: order.model_copy(deep=True) for order in orders or []}
        self.cancel_count = 0

    @classmethod
    def with_sample_orders(cls) -> "OrderRepository":
        """Create the deterministic dataset used by exercises and tests."""
        return cls(
            [
                Order(order_id="ORDER-1001", status=OrderStatus.PENDING),
                Order(order_id="ORDER-1002", status=OrderStatus.SHIPPED),
                Order(order_id="ORDER-1003", status=OrderStatus.CANCELLED),
            ]
        )

    def query(self, order_id: str) -> OrderQueryResult:
        """Return a business result instead of leaking storage implementation."""
        order = self._orders.get(order_id)
        if not order:
            return OrderQueryResult(order_id=order_id, success=False, error_code='ORDER_NOT_FOUND', message='Order not found')

        return OrderQueryResult(order_id=order_id, success=True, status=order.status, message='success')


    def cancel(self, order_id: str, reason: str) -> CancelOrderResult:
        """Cancel one eligible order exactly once."""

        if not reason or reason.strip() == "":
            return CancelOrderResult(order_id=order_id, success=False, error_code='REASON_REQUIRED', message='Reason is required')

        order = self._orders.get(order_id)
        if not order:
            return CancelOrderResult(order_id=order_id, success=False, error_code='ORDER_NOT_FOUND', message='Order not found')
        if order.status == OrderStatus.CANCELLED:
            return CancelOrderResult(order_id=order_id, success=False, error_code='ORDER_ALREADY_CANCELLED', message='Order already canceled')
        if order.status == OrderStatus.SHIPPED:
            return CancelOrderResult(order_id=order_id, success=False, error_code='ORDER_ALREADY_SHIPPED', message='Order already shipped')
        # with distinct error codes.

        previous_status = order.status
        self._orders[order_id].status = OrderStatus.CANCELLED
        self.cancel_count += 1
        return CancelOrderResult(order_id=order_id, success=True, previous_status=previous_status, current_status=OrderStatus.CANCELLED, message='Order cancelled')


