import asyncio
import os
import time
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import base58

from exchanges.bulk import BulkClient


def _resting(order_id: str) -> dict:
    return {"response": {"data": {"statuses": [{"resting": {"oid": order_id}}]}}}


class BulkClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.key = base58.b58encode(bytes(range(32))).decode()
        self.environment = patch.dict(os.environ, {"BULK_PRIVATE_KEY": self.key}, clear=False)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.config = type("Config", (), {
            "ticker": "ETH", "contract_id": "", "tick_size": Decimal(0),
            "quantity": Decimal("0.1"), "direction": "buy",
        })()
        self.client = BulkClient(self.config)

    def _market_is_connected(self):
        self.client._ws_task = asyncio.get_running_loop().create_future()
        self.client._best_bid = Decimal("1999")
        self.client._best_ask = Decimal("2000")
        self.client._book_update_time = time.monotonic()

    def test_mainnet_is_the_default_signature_domain(self):
        self.assertEqual(self.client.api_url, BulkClient.API_URL)
        self.assertEqual(self.client.ws_url, BulkClient.WS_URL)
        self.assertEqual(self.client.signature_domain, "mainnet")
        subscription = self.client.subscription_payload()["subscription"]
        self.assertEqual(subscription[0], {"type": "account", "user": self.client.public_key})
        self.assertEqual(subscription[1]["type"], "l2Snapshot")
        self.assertEqual(subscription[1]["symbol"], "ETH-USD")

    def test_testnet_endpoints_sign_for_testnet(self):
        with patch.dict(os.environ, {
            "BULK_PRIVATE_KEY": self.key,
            "BULK_API_URL": BulkClient.TESTNET_API_URL,
            "BULK_WS_URL": BulkClient.TESTNET_WS_URL,
        }):
            client = BulkClient(self.config)
        self.assertEqual(client.signature_domain, "testnet")

    def test_keychain_signs_an_alo_limit_for_mainnet(self):
        signed = self.client._signer.sign({
            "type": "order",
            "symbol": "ETH-USD",
            "is_buy": True,
            "price": 2000.0,
            "size": 0.1,
            "reduce_only": False,
            "order_type": {"type": "limit", "tif": "ALO"},
        })
        self.assertEqual(signed["account"], self.client.public_key)
        self.assertEqual(signed["signer"], self.client.public_key)
        self.assertTrue(signed["signature"])
        self.assertEqual(signed["actions"][0]["l"]["tif"], "ALO")
        self.assertFalse(signed["actions"][0]["l"]["r"])

    def test_mixed_networks_are_rejected(self):
        with patch.dict(os.environ, {
            "BULK_PRIVATE_KEY": self.key,
            "BULK_API_URL": BulkClient.API_URL,
            "BULK_WS_URL": BulkClient.TESTNET_WS_URL,
        }):
            with self.assertRaisesRegex(ValueError, "different networks"):
                BulkClient(self.config)

    async def test_connect_waits_for_account_and_book(self):
        self.client._account = AsyncMock(return_value={"openOrders": [], "positions": []})

        async def socket():
            self.client._handle_message({
                "type": "l2Snapshot",
                "data": {"book": {
                    "symbol": "ETH-USD",
                    "levels": [[{"px": "1999", "sz": "1"}], [{"px": "2000", "sz": "1"}]],
                }},
            })
            self.client._handle_message({"type": "account", "data": {"type": "snapshot"}})
            await asyncio.Event().wait()

        with patch.object(self.client, "_run_market_socket", socket):
            await self.client.connect()
        self.assertEqual(await self.client.fetch_bbo_prices("ETH-USD"), (Decimal("1999"), Decimal("2000")))
        self.client._ws_task.cancel()

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
        self._market_is_connected()
        self.client._lot_size = Decimal("0.0001")
        self.client._min_notional = Decimal("50")
        self.config.tick_size = Decimal("0.001")
        signed = []

        class StubSigner:
            def sign(self, payload):
                signed.append(payload)
                order_id = "entry" if len(signed) == 1 else "exit"
                return {
                    "actions": [{"l": {"tif": "ALO"}}], "nonce": 1,
                    "account": "a", "signer": "a", "signature": "s", "order_id": order_id,
                }

        self.client._signer = StubSigner()
        self.client._post_signed = AsyncMock(side_effect=[_resting("entry"), _resting("exit")])
        entry = await self.client.place_open_order("ETH-USD", Decimal("0.1"), "buy")
        close = await self.client.place_close_order("ETH-USD", Decimal("0.1"), Decimal("2001"), "sell")
        self.assertTrue(entry.success and close.success)
        self.assertEqual(entry.price, Decimal("1999.999"))
        self.assertEqual(signed[0]["order_type"], {"type": "limit", "tif": "ALO"})
        self.assertFalse(signed[0]["reduce_only"])
        self.assertTrue(signed[1]["is_buy"] is False)
        self.assertTrue(signed[1]["reduce_only"])
        posted = self.client._post_signed.await_args_list[0].args[0]
        self.assertEqual(set(posted), {"actions", "nonce", "account", "signer", "signature", "order_id"})

    async def test_cancel_returns_confirmed_partial_fill(self):
        self._market_is_connected()
        self.client._orders["entry"] = self.client._parse_order({
            "sym": "ETH-USD", "oid": "entry", "sz": 1, "origSz": 1,
            "fillSz": 0, "px": 2000, "status": "placed", "isBuy": True,
        })
        self.client._account = AsyncMock(return_value={
            "openOrders": [{"sym": "ETH-USD", "oid": "entry", "sz": 1,
                            "origSz": 1, "fillSz": 0, "px": 2000, "status": "placed", "isBuy": True}],
            "positions": [],
        })
        events = []
        self.client.setup_order_update_handler(events.append)

        async def post(_signed):
            self.client._handle_account_data({
                "type": "orderUpdate",
                "sym": "ETH-USD",
                "oid": "entry",
                "isBuy": True,
                "sz": "0.6",
                "origSz": "1",
                "fillSz": "0.4",
                "px": "2000",
                "status": "cancelled",
            })
            return {"response": {"data": {"statuses": [{"cancelled": {"oid": "entry"}}]}}}

        self.client._signer = type("StubSigner", (), {
            "sign": staticmethod(lambda payload: {
                "actions": [{"cx": {}}], "nonce": 1, "account": "a",
                "signer": "a", "signature": "s",
            }),
        })()
        self.client._post_signed = post
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
