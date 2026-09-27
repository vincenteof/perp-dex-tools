import asyncio
import json
import os
import unittest
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from exchanges.arcus import ArcusAPIError, ArcusClient
from exchanges.factory import ExchangeFactory
from tests import test_bulk_bot
import aiohttp


class ArcusBotOrderEventTests(test_bulk_bot.BulkBotOrderEventTests):
    """Same strategy/event guarantees as the existing Bulk path."""
    def setUp(self):
        super().setUp()
        self.bot.config.exchange = "arcus"

    async def test_unknown_submit_stops_bot_instead_of_opening_again(self):
        self.bot.exchange_client.place_open_order = AsyncMock(
            side_effect=RuntimeError("Arcus submit outcome is unknown"))
        with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
            await self.bot._place_and_monitor_open_order()


class ArcusTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "ARCUS_API_SIGNING_KEY": "01" * 32, "ARCUS_ADDRESS": "0x" + "AB" * 20,
            "ARCUS_NETWORK": "testnet", "ARCUS_ACCOUNT_INDEX": "3", "ARCUS_API_KEY": "",
        })
        self.env.start()
        self.addCleanup(self.env.stop)
        self.config = SimpleNamespace(ticker="ETH", quantity=Decimal("0.02"),
                                      tick_size=Decimal("0.01"), direction="buy", boost_mode=False)
        self.client = ArcusClient(self.config)
        self.client.market_id = 2
        self.client.tick, self.client.step = Decimal("0.01"), Decimal("0.0000001")
        self.client.market = {
            "marketDisplayName": "ETH-USD", "marketId": 2, "status": "ONLINE",
            "tickSize": "0.01", "stepSize": "0.0000001", "minOrderNotional": "5",
            "minOrderSize": "0.001", "maxOrderSize": "100000",
            "tickTiers": [{"upToPrice": "10000", "tick": "0.01"}, {"tick": "0.05"}],
        }
        self.events = []
        self.client.setup_order_update_handler(self.events.append)

    def raw(self, oid="entry", status="OPEN", remaining="0.02", seq=1, **extra):
        return {"orderId": oid, "marketId": 2, "marketDisplayName": "ETH-USD", "side": "BUY",
                "status": status, "originalSize": "0.02", "remainingSize": remaining,
                "price": "2700.01", "updatedAt": seq * 1000, "sequenceNumber": seq, **extra}

    async def test_factory_and_network(self):
        self.assertIn("arcus", ExchangeFactory.get_supported_exchanges())
        self.assertEqual(self.client.api_url, "https://api.testnet.arcus.xyz/v1")
        self.assertEqual(self.client.address, "0x" + "ab" * 20)
        self.assertEqual(self.client._scope()["accountIndex"], 3)

    async def test_config_rejects_wallet_key_and_mismatched_api_key(self):
        with patch.dict(os.environ, {"ARCUS_API_SIGNING_KEY": "not-an-api-key"}):
            with self.assertRaisesRegex(ValueError, "32-byte hex"):
                ArcusClient(self.config)
        with patch.dict(os.environ, {"ARCUS_API_KEY": "ff" * 32}):
            with self.assertRaisesRegex(ValueError, "does not match"):
                ArcusClient(self.config)
        with patch.dict(os.environ, {"ARCUS_ACCOUNT_INDEX": "10"}):
            with self.assertRaises(ValueError):
                ArcusClient(self.config)

    async def test_typed_signature_uses_base_tick_and_ns_not_tier_tick(self):
        payload = self.client._place_payload(Decimal("0.02"), Decimal("10000.05"),
                                              "sell", True, "Case-Sensitive", 123456789, 456789)
        expected = {"ad": "0x" + "ab" * 20, "ai": 3, "c": "Case-Sensitive", "ct": 123456789,
                    "g": 456789000, "m": 2, "op": 1, "p": 1000005, "q": 200000,
                    "r": 1, "s": 1, "t": 3, "v": 1}
        self.assertEqual(payload, expected)
        headers = self.client._headers(payload, 123456789)
        self.client._signer.public_key().verify(bytes.fromhex(headers["X-Signature"]),
            json.dumps(expected, sort_keys=True, separators=(",", ":")).encode())
        self.assertEqual(headers["X-Timestamp"], "123456789")
        with self.assertRaises(ValueError):
            self.client._units(Decimal("0.011"), Decimal("0.01"))

    async def test_tier_rounding_and_boundary(self):
        self.assertEqual(self.client._snap(Decimal("10000.023"), "sell"), Decimal("10000.05"))
        self.assertEqual(self.client._snap(Decimal("9999.999"), "sell"), Decimal("10000"))
        self.assertEqual(self.client._snap(Decimal("10000.023"), "buy"), Decimal("10000"))

    async def test_market_validation(self):
        self.client._get = AsyncMock(return_value={"markets": [self.client.market]})
        self.assertEqual(await self.client.get_contract_attributes(), ("ETH-USD", Decimal("0.01")))
        self.config.quantity = Decimal("0.00000011")
        with self.assertRaisesRegex(ValueError, "multiple"):
            await self.client.get_contract_attributes()

    async def test_maker_ack_waits_for_order_and_early_fill(self):
        async def submit(_method, _path, **kwargs):
            body = kwargs["json"]
            self.assertEqual(body["timeInForce"], "ALO")
            self.assertEqual(body["accountIndex"], 3)
            self.assertFalse(body["reduceOnly"])
            self.assertGreater(int(body["goodTilTime"]), body["timestamp"] // 1000 + 31 * 86400 * 10**6)
            self.client._remember(self.raw("new", "FILLED", "0", clientId=body["clientId"]))
            return {"orderId": "new", "status": "ACK"}
        self.client._request = submit
        result = await self.client._place(Decimal("0.02"), Decimal("2700.01"), "buy", False)
        self.assertEqual(result.status, "FILLED")
        self.assertEqual(result.filled_size, Decimal("0.02"))
        self.assertEqual(len(self.events), 1)

    async def test_ack_is_not_success_until_confirmed(self):
        self.client._request = AsyncMock(return_value={"orderId": "new", "status": "ACK"})
        self.client._confirm = AsyncMock(side_effect=ValueError("bad state"))
        with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
            await self.client._place(Decimal("0.02"), Decimal("2700.01"), "buy", False)
        self.client._request.assert_awaited_once()

    async def test_submit_timeout_not_retried(self):
        self.client._request = AsyncMock(side_effect=TimeoutError())
        with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
            await self.client._place(Decimal("0.02"), Decimal("2700.01"), "buy", False)
        self.client._request.assert_awaited_once()

    async def test_rejection_and_reduce_only_dust(self):
        self.client._request = AsyncMock(side_effect=ArcusAPIError(403, "Forbidden"))
        result = await self.client._place(Decimal("0.02"), Decimal("2700.01"), "buy", False)
        self.assertFalse(result.success)
        with self.assertRaisesRegex(ValueError, "minimum opening"):
            await self.client._place(Decimal("0.0001"), Decimal("2700.01"), "buy", False)
        self.client._request = AsyncMock(return_value={"orderId": "close", "status": "ACK"})
        order = self.client._remember(self.raw("close", side="SELL"))
        self.client._confirm = AsyncMock(return_value=order)
        await self.client._place(Decimal("0.0001"), Decimal("2700.01"), "sell", True)
        self.assertTrue(self.client._request.await_args.kwargs["json"]["reduceOnly"])

    async def test_snapshot_does_not_replay_historical_fills(self):
        await self.client._reconcile({"openOrders": [self.raw()],
                                     "recentClosedOrders": [self.raw("old", "FILLED", "0")]})
        self.assertEqual(self.events, [])
        self.assertEqual(self.client._orders["old"].status, "FILLED")

    async def test_duplicates_stale_frames_and_terminal_regression(self):
        self.client._tracked.add("entry")
        self.client._remember(self.raw(remaining="0.01", seq=2))
        self.client._remember(self.raw(remaining="0.01", seq=2))
        self.client._remember(self.raw(seq=1))
        self.assertEqual(len(self.events), 1)
        self.client._remember(self.raw(status="FILLED", remaining="0", seq=3))
        self.client._remember(self.raw(seq=4))
        self.assertEqual(self.client._orders["entry"].status, "FILLED")
        self.assertEqual(len(self.events), 2)

    async def test_reconnect_fetches_order_missing_from_snapshot(self):
        self.client._tracked.add("entry")
        self.client._remember(self.raw())
        self.client._get = AsyncMock(return_value=self.raw(status="FILLED", remaining="0", seq=5))
        await self.client._reconcile({"openOrders": [], "recentClosedOrders": []})
        self.assertEqual(self.client._orders["entry"].status, "FILLED")
        self.assertEqual(self.events[-1]["filled_size"], "0.02")

    async def test_reconnect_missing_order_fails_closed(self):
        self.client._tracked.add("missing")
        self.client._get = AsyncMock(side_effect=ArcusAPIError(404, "not found"))
        with self.assertRaisesRegex(ValueError, "Cannot reconcile"):
            await self.client._reconcile({"openOrders": [], "recentClosedOrders": []})

    async def test_cancel_ack_requires_terminal_and_preserves_partial_fill(self):
        self.client._remember(self.raw(remaining="0.015"))
        self.client._ready.set()
        self.client._request = AsyncMock(return_value={"status": "CANCEL_ACKNOWLEDGED"})
        final = self.client._remember(self.raw(status="CANCELED", remaining="0.015", seq=2))
        self.client._confirm = AsyncMock(return_value=final)
        # Restore live state to exercise submit instead of cached terminal shortcut.
        self.client._orders["entry"].status = "OPEN"
        result = await self.client.cancel_order("entry")
        self.client._confirm.assert_awaited_once()
        self.assertEqual(result.filled_size, Decimal("0.005"))
        self.assertEqual(self.client._request.await_args.kwargs["json"]["kind"], "orderId")

    async def test_cancel_unknown_never_claims_canceled(self):
        self.client._remember(self.raw())
        self.client._ready.set()
        self.client._request = AsyncMock(side_effect=TimeoutError())
        self.client._confirm = AsyncMock(side_effect=TimeoutError())
        with self.assertRaisesRegex(RuntimeError, "cancel outcome is unknown"):
            await self.client.cancel_order("entry")

    async def test_tracked_order_query_failure_is_unknown_not_missing(self):
        self.client._tracked.add("entry")
        self.client._get = AsyncMock(side_effect=ArcusAPIError(404, "not found"))
        with self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
            await self.client.get_order_info("entry")

    async def test_remaining_orders_and_short_position(self):
        self.client._get = AsyncMock(return_value={"orders": [self.raw(remaining="0.01", side="SELL")]})
        result = await self.client.get_active_orders("ETH-USD")
        self.assertEqual(result[0].remaining_size, Decimal("0.01"))
        self.client._get = AsyncMock(return_value={"positions": {"2": {"size": "-0.12"}}})
        self.assertEqual(await self.client.get_account_positions(), Decimal("-0.12"))
        self.client._get.return_value = {"positions": {}}
        self.assertEqual(await self.client.get_account_positions(), Decimal(0))

    async def test_startup_refuses_unmanaged_opening_orders(self):
        self.client._validate_api_key = AsyncMock()
        self.client._get = AsyncMock(return_value={"equity": "100"})
        self.client.get_active_orders = AsyncMock(return_value=[self.client._remember(self.raw())])
        with self.assertRaisesRegex(ValueError, "pre-existing opening"):
            await self.client.connect()

    async def test_key_registration_scope_and_expiry_checked_without_wallet_secret(self):
        key = {"apiKey": self.client.api_key, "status": "ACTIVE", "accountIndex": 3,
               "validUntil": 0}
        self.client._get = AsyncMock(return_value={"apiKeys": [key]})
        await self.client._validate_api_key()
        key["accountIndex"] = 2
        with self.assertRaisesRegex(ValueError, "ACCOUNT_INDEX"):
            await self.client._validate_api_key()
        key["allSubaccounts"] = True
        await self.client._validate_api_key()
        key["validUntil"] = 1
        with self.assertRaisesRegex(ValueError, "expired"):
            await self.client._validate_api_key()
        key["validUntil"], key["status"] = 0, "DELETED"
        with self.assertRaisesRegex(ValueError, "registered/active"):
            await self.client._validate_api_key()

    async def test_fatal_recovery_wakes_waiters(self):
        self.client._fatal = "missing order"
        with self.assertRaisesRegex(RuntimeError, "recovery needs attention"):
            await self.client._wait_ready()

    async def test_socket_reconnect_resubscribes_and_reconciles_before_ready(self):
        client = self.client
        client._tracked.add("entry")
        client._get = AsyncMock(return_value=self.raw(status="FILLED", remaining="0", seq=5))
        subscriptions, readiness = [], []

        class Socket:
            def __init__(self, number):
                self.number = number

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                pass

            async def send_json(self, data):
                subscriptions.append(data)

            def __aiter__(self):
                return self.frames()

            async def frames(self):
                messages = [
                    {"type": "subscribed", "channel": "l2Orderbook", "id": "ETH-USD",
                     "contents": {"bids": [["2700", "1"]], "asks": [["2700.01", "1"]]}},
                    {"type": "subscribed", "channel": "orders", "id": client.address,
                     "accountIndex": 3, "contents": {"isSnapshot": True,
                       "openOrders": [self_test.raw()] if self.number == 1 else [],
                       "recentClosedOrders": []}},
                ]
                for data in messages:
                    yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(data))
                readiness.append(client._ready.is_set())
                if self.number == 2:
                    client._stop.set()

        self_test = self
        count = 0

        def connect(*_args, **_kwargs):
            nonlocal count
            count += 1
            return Socket(count)

        client._ws_http = SimpleNamespace(closed=False, ws_connect=connect)
        with patch("exchanges.arcus.asyncio.sleep", new=AsyncMock()):
            await client._socket_loop()
        self.assertEqual(readiness, [True, True])
        self.assertEqual(len(subscriptions), 4)
        self.assertEqual(subscriptions[1]["accountIndex"], 3)
        self.assertEqual(subscriptions[1]["market"], "ETH-USD")
        self.assertEqual(subscriptions[1], subscriptions[3])
        self.assertEqual(client._orders["entry"].status, "FILLED")
        self.assertEqual(sum(e["status"] == "FILLED" for e in self.events), 1)


if __name__ == "__main__":
    unittest.main()
