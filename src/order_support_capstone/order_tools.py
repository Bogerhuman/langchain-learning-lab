"""LangChain Tool adapters around the order business layer."""

from langchain_core.tools import BaseTool, tool

from order_support_capstone.repository import OrderRepository


def build_order_tools(repository: OrderRepository) -> list[BaseTool]:
    """Bind one repository instance to query and cancellation tools."""

    @tool
    def query_order(order_id: str) -> dict:
        """Query the current status of one order by its exact order ID."""

        result = repository.query(order_id)

        return result.model_dump(mode="json")

    @tool
    def cancel_order(order_id: str, reason: str) -> dict:
        """Cancel an eligible order for a stated reason after required approval."""
        # Keep all business validation in the repository, not in this adapter.
        result = repository.cancel(order_id, reason)
        return result.model_dump(mode="json")

    return [query_order, cancel_order]
