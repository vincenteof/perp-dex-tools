import os
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import base58
from bulk_api.common import OrderStatus, Side, TimeInForce
from bulk_api.messages.trade import OrderResponse

from exchanges.bulk import BulkClient


class BulkClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.key = base58.b58encode(bytes(range(32))).decode()
        self.environment = patch.dict(os.environ, {"BULK_PRIVATE_KEY": self.key})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.config = SimpleNamespace(
            ticker="ETH", contract_id="", tick_size=Decimal(0),
            quantity=Decimal("0.1"), direction="buy",
        )
        self.client = BulkClient(self.config)

    async def test_market_metadata_validates_lot_and_uses_actual_api_list(self):
        self.client._request = AsyncMock(return_value=[{
            "symbol": "ETH-USD", "status": "TRADING", "tickSize": 0.001,
            "lotSize": 0.0001, "minNotional": 50.0,
        }])
        self.assertEqual(await self.client.get_contract_attributes(), ("ETH-USD", Decimal("0.001")))
        self.config.quantity = Decimal("0.10001")
        with self.assertRaisesRegex(ValueError, "multiple"):
            await self.client.get_contract_attributes()

    async def test_entry_and_close_are_post_only_and_close_is_reduce_only(self):
        self.client._lot_size = Decimal("0.0001")
        self.client._min_notional = Decimal("50")
        self.client.ws = SimpleNamespace(
            is_connected=True,
            place_limit_order=AsyncMock(side_effect=[
                OrderResponse("entry", OrderStatus.RESTING, None, {}),
                OrderResponse("exit", OrderStatus.RESTING, None, {}),
            ]),
        )
        self.client.fetch_bbo_prices = AsyncMock(return_value=(Decimal("1999"), Decimal("2000")))
        self.config.tick_size = Decimal("0.001")
        entry = await self.client.place_open_order("ETH-USD", Decimal("0.1"), "buy")
        close = await self.client.place_close_order("ETH-USD", Decimal("0.1"), Decimal("2001"), "sell")
        self.assertTrue(entry.success and close.success)
        self.assertEqual(entry.price, Decimal("1999.999"))
        self.assertEqual(self.client.ws.place_limit_order.await_args_list[0].kwargs["time_in_force"], TimeInForce.ALO)
        self.assertFalse(self.client.ws.place_limit_order.await_args_list[0].kwargs["reduce_only"])
        self.assertTrue(self.client.ws.place_limit_order.await_args_list[1].kwargs["reduce_only"])

    async def test_cancel_returns_confirmed_partial_fill(self):
        self.client._orders["entry"] = self.client._parse_order({
            "sym": "ETH-USD", "oid": "entry", "sz": 1, "origSz": 1,
            "fillSz": 0, "px": 2000, "status": "placed",
        })
        self.client._account = AsyncMock(return_value={
            "openOrders": [{"sym": "ETH-USD", "oid": "entry", "sz": 1,
                            "origSz": 1, "fillSz": 0, "px": 2000, "status": "placed"}],
            "positions": [],
        })
        events = []
        self.client.setup_order_update_handler(events.append)

        async def cancel(**_kwargs):
            self.client._on_order(SimpleNamespace(
                symbol="ETH-USD", order_id="entry", side=Side.BUY,
                status=OrderStatus.CANCELLED, size=Decimal("0.6"),
                size_orig=Decimal("1"), size_done=Decimal("0.4"), price=2000,
            ))
            return OrderResponse(None, OrderStatus.CANCELLED, None, {})

        self.client.ws = SimpleNamespace(is_connected=True, cancel_order=AsyncMock(side_effect=cancel))
        result = await self.client.cancel_order("entry")
        self.assertTrue(result.success)
        self.assertEqual(result.filled_size, Decimal("0.4"))
        self.assertEqual(events[-1]["status"], "CANCELED")
        self.assertEqual(events[-1]["filled_size"], "0.4")

    async def test_account_read_failure_does_not_look_like_empty_position(self):
        self.client._request = AsyncMock(return_value={"error": "unavailable"})
        with self.assertRaisesRegex(ValueError, "invalid account"):
            await self.client.get_account_positions()

    async def test_live_orders_expose_remaining_size_and_signed_position(self):
        self.client._request = AsyncMock(return_value=[{"fullAccount": {
            "openOrders": [{
                "oid": "close", "sym": "ETH-USD", "sz": -0.06,
                "origSz": 0.1, "fillSz": 0.04, "px": 2001,
                "status": "partiallyFilled",
            }],
            "positions": [{"symbol": "ETH-USD", "size": -0.06}],
        }}])
        orders = await self.client.get_active_orders("ETH-USD")
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0].side, "sell")
        self.assertEqual(orders[0].status, "PARTIALLY_FILLED")
        self.assertEqual(orders[0].remaining_size, Decimal("0.06"))
        self.assertEqual(await self.client.get_account_positions(), Decimal("-0.06"))


if __name__ == "__main__":
    unittest.main()
