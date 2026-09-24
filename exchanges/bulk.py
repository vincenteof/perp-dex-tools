"""BULK perpetual exchange adapter for the single-exchange trading bot.

Orders are signed with bulk-keychain and submitted with HTTP POST /order, the
same path used by the working strategy. The WebSocket is only the public book
and the account stream.
"""

import asyncio
import json
import os
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from bulk_keychain import Signer

from .base import BaseExchangeClient, OrderInfo, OrderResult


class BulkClient(BaseExchangeClient):
    # Website accounts live on mainnet. The exchange-api / exchange-ws1 hosts are testnet.
    API_URL = "https://mainnet-api1.bulk.trade/api/v1"
    WS_URL = "wss://mainnet-ws1.bulk.trade"
    TESTNET_API_URL = "https://exchange-api.bulk.trade/api/v1"
    TESTNET_WS_URL = "wss://exchange-ws1.bulk.trade"

    def __init__(self, config):
        super().__init__(config)
        self.symbol = f"{config.ticker.upper()}-USD"
        self.api_url = os.getenv("BULK_API_URL", self.API_URL).rstrip("/")
        self.ws_url = os.getenv("BULK_WS_URL", self.WS_URL)
        self.signature_domain = self.resolve_signature_domain(self.api_url, self.ws_url)
        self._signer = Signer.from_base58(os.environ["BULK_PRIVATE_KEY"], self.signature_domain)
        self._signer.set_compute_order_id(True)
        expected_account = os.getenv("BULK_ACCOUNT", "").strip()
        if expected_account and expected_account != self._signer.pubkey:
            raise ValueError("Bulk private key does not match BULK_ACCOUNT")
        self._connect_error: Optional[str] = None
        self._account_seen = False
        self._best_bid: Optional[Decimal] = None
        self._best_ask: Optional[Decimal] = None
        self._ws_task: Optional[asyncio.Task] = None
        self._ws_stop = asyncio.Event()
        self.http: Optional[aiohttp.ClientSession] = None
        self._handler = None
        self._orders: Dict[str, OrderInfo] = {}
        self._order_events: Dict[str, asyncio.Event] = {}
        self._fill_totals: Dict[str, Decimal] = {}
        self._book_update_time = 0.0
        self._lot_size = Decimal("0")
        self._min_notional = Decimal("0")

    def _validate_config(self) -> None:
        if not os.getenv("BULK_PRIVATE_KEY"):
            raise ValueError("BULK_PRIVATE_KEY is required for Bulk trading")

    def get_exchange_name(self) -> str:
        return "bulk"

    @property
    def public_key(self) -> str:
        return self._signer.pubkey

    @staticmethod
    def resolve_signature_domain(api_url: str, ws_url: str) -> str:
        """Return the bulk-keychain network name shared by both endpoints."""
        explicit = os.getenv("BULK_SIGNATURE_DOMAIN", "").strip().lower()
        if explicit:
            if explicit not in {"mainnet", "testnet", "devnet"}:
                raise ValueError("BULK_SIGNATURE_DOMAIN must be mainnet, testnet, or devnet")
            return explicit
        api_domain = BulkClient._domain_for_endpoint(api_url)
        ws_domain = BulkClient._domain_for_endpoint(ws_url)
        if api_domain != ws_domain:
            raise ValueError(
                f"Bulk API ({api_url}) and WebSocket ({ws_url}) are on different networks"
            )
        return api_domain

    @staticmethod
    def _domain_for_endpoint(url: str) -> str:
        host = url.lower()
        if "mainnet" in host:
            return "mainnet"
        if "devnet" in host:
            return "devnet"
        if ("exchange-api.bulk.trade" in host or "exchange-ws1.bulk.trade" in host
                or "testnet" in host):
            return "testnet"
        raise ValueError(
            f"Set BULK_SIGNATURE_DOMAIN to mainnet, testnet, or devnet for Bulk endpoint {url}"
        )

    def setup_order_update_handler(self, handler) -> None:
        self._handler = handler

    def subscription_payload(self) -> Dict[str, Any]:
        return {
            "method": "subscribe",
            "subscription": [
                {"type": "account", "user": self.public_key},
                {
                    "type": "l2Snapshot",
                    "symbol": self.symbol,
                    "nlevels": 20,
                    "aggregation": 0,
                },
            ],
        }

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        if self.http is None:
            self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        async with self.http.request(method, f"{self.api_url}/{path}", **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def _account(self) -> Dict[str, Any]:
        data = await self._request("POST", "account", json={
            "type": "fullAccount", "user": self.public_key,
        })
        if not isinstance(data, list) or not data or not isinstance(data[0], dict):
            raise ValueError("Bulk returned an invalid account response")
        account = data[0].get("fullAccount")
        if (not isinstance(account, dict)
                or not isinstance(account.get("openOrders"), list)
                or not isinstance(account.get("positions"), list)):
            raise ValueError("Bulk account response is missing orders or positions")
        return account

    async def get_contract_attributes(self) -> Tuple[str, Decimal]:
        data = await self._request("GET", "exchangeInfo")
        markets = data if isinstance(data, list) else data.get("markets", {})
        if isinstance(markets, dict):
            market = markets.get(self.symbol)
        else:
            market = next((item for item in markets if item.get("symbol") == self.symbol), None)
        if not market or market.get("status") != "TRADING":
            raise ValueError(f"Bulk market {self.symbol} is unavailable")
        tick = Decimal(str(market["tickSize"]))
        self._lot_size = Decimal(str(market["lotSize"]))
        self._min_notional = Decimal(str(market.get("minNotional", 0)))
        quantity = self.config.quantity
        if tick <= 0 or self._lot_size <= 0 or quantity <= 0:
            raise ValueError("Bulk tick, lot, and order quantity must be positive")
        if quantity % self._lot_size != 0:
            raise ValueError(f"Bulk quantity {quantity} must be a multiple of {self._lot_size}")
        self.config.contract_id = self.symbol
        self.config.tick_size = tick
        return self.symbol, tick

    async def connect(self) -> None:
        try:
            await self._account()
        except ValueError as exc:
            raise ConnectionError(
                f"Bulk account {self.public_key} cannot be read on {self.api_url} "
                f"({self.signature_domain}). Mainnet is {self.API_URL} and {self.WS_URL}; "
                f"testnet is {self.TESTNET_API_URL} and {self.TESTNET_WS_URL}. "
                f"Point BULK_API_URL and BULK_WS_URL at the same network. {exc}"
            ) from exc
        self._connect_error = None
        self._account_seen = False
        self._ws_stop = asyncio.Event()
        self._ws_task = asyncio.create_task(self._run_market_socket())
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if self._connect_error:
                raise ConnectionError(
                    f"Bulk WebSocket on {self.ws_url} ({self.signature_domain}) rejected "
                    f"account {self.public_key}: {self._connect_error}"
                )
            if (self._account_seen and self._book_update_time
                    and self._best_bid and self._best_ask):
                return
            await asyncio.sleep(0.1)
        raise TimeoutError(
            f"Bulk account {self.public_key} or the {self.symbol} order book "
            f"did not arrive from {self.ws_url} ({self.signature_domain})"
        )

    async def _run_market_socket(self) -> None:
        if self.http is None:
            self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        try:
            async with self.http.ws_connect(self.ws_url, heartbeat=20) as websocket:
                await websocket.send_json(self.subscription_payload())
                while not self._ws_stop.is_set():
                    message = await websocket.receive()
                    if message.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING}:
                        break
                    if message.type == aiohttp.WSMsgType.ERROR:
                        self._connect_error = "Bulk WebSocket connection failed"
                        break
                    if message.type != aiohttp.WSMsgType.TEXT:
                        continue
                    self._handle_message(json.loads(message.data))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._connect_error is None:
                self._connect_error = str(exc)

    def _handle_message(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("type") == "error" or payload.get("error"):
            error = payload.get("error")
            message = error.get("message") if isinstance(error, dict) else payload.get("message")
            self._connect_error = str(message or error or "websocket error")
            return
        if payload.get("type") == "l2Snapshot":
            self._handle_book(payload)
            return
        if payload.get("type") == "account":
            self._account_seen = True
            data = payload.get("data")
            if isinstance(data, dict):
                self._handle_account_data(data)

    def _handle_book(self, payload: Dict[str, Any]) -> None:
        data = payload.get("data")
        book = data.get("book") if isinstance(data, dict) else None
        if not isinstance(book, dict) or book.get("symbol") != self.symbol:
            return
        levels = book.get("levels")
        if not isinstance(levels, list) or len(levels) != 2:
            return
        bids, asks = levels
        if not bids or not asks:
            return
        bid = Decimal(str(bids[0]["px"]))
        ask = Decimal(str(asks[0]["px"]))
        if bid <= 0 or ask <= bid:
            return
        self._best_bid = bid
        self._best_ask = ask
        self._book_update_time = time.monotonic()

    def _handle_account_data(self, data: Dict[str, Any]) -> None:
        kind = str(data.get("type") or "")
        if kind == "orderUpdate":
            self._remember(self._parse_order(data))
            return
        if kind == "fill":
            self._apply_fill(data)
            return
        snapshot = data.get("fullAccount") if isinstance(data.get("fullAccount"), dict) else data
        if isinstance(snapshot, dict) and isinstance(snapshot.get("openOrders"), list):
            for raw in snapshot["openOrders"]:
                if isinstance(raw, dict):
                    self._remember(self._parse_order(raw))

    async def disconnect(self) -> None:
        self._ws_stop.set()
        task = self._ws_task
        self._ws_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self.http is not None:
            await self.http.close()
            self.http = None

    @staticmethod
    def _status(status: Any) -> str:
        name = getattr(status, "name", str(status)).upper().replace("-", "").replace("_", "")
        if name in {"RESTING", "WORKING", "CANCELPENDING", "PLACED", "OPEN", "NEW"}:
            return "OPEN"
        if name in {"PARTIALLYFILLED", "PARTIAL"}:
            return "PARTIALLY_FILLED"
        if name.startswith("CANCELLED") or name.startswith("CANCELED"):
            return "CANCELED"
        if name in {"FILLED", "FILL"}:
            return "FILLED"
        return name

    @staticmethod
    def _side_name(is_buy: Any, signed_size: Decimal) -> str:
        if is_buy is True or str(is_buy).lower() in {"buy", "bid", "true"}:
            return "buy"
        if is_buy is False or str(is_buy).lower() in {"sell", "ask", "false"}:
            return "sell"
        return "buy" if signed_size >= 0 else "sell"

    def _remember(self, order: OrderInfo) -> None:
        prior = self._orders.get(order.order_id)
        filled = max(order.filled_size, self._fill_totals.get(order.order_id, Decimal(0)))
        if prior is not None:
            filled = max(filled, prior.filled_size)
        order.filled_size = filled
        order.remaining_size = max(order.size - filled, Decimal(0))
        if order.remaining_size == 0 and order.status != "CANCELED":
            order.status = "FILLED"
        self._orders[order.order_id] = order
        self._emit(order)
        if order.status in {"FILLED", "CANCELED"}:
            self._order_events.setdefault(order.order_id, asyncio.Event()).set()

    def _apply_fill(self, fill: Dict[str, Any]) -> None:
        symbol = fill.get("symbol") or fill.get("sym")
        order_id = str(fill.get("orderId") or fill.get("oid") or "")
        if symbol not in {None, self.symbol} or not order_id:
            return
        size = Decimal(str(fill.get("size") if fill.get("size") is not None else fill.get("amount") or 0))
        total = self._fill_totals.get(order_id, Decimal(0)) + size
        self._fill_totals[order_id] = total
        order = self._orders.get(order_id)
        if order is None:
            return
        order.filled_size = max(order.filled_size, total)
        order.remaining_size = max(order.size - order.filled_size, Decimal(0))
        if order.remaining_size == 0:
            order.status = "FILLED"
            self._order_events.setdefault(order.order_id, asyncio.Event()).set()
        elif order.status != "CANCELED":
            order.status = "PARTIALLY_FILLED"
        self._emit(order)

    def _emit(self, order: OrderInfo) -> None:
        if self._handler is not None:
            self._handler({
                "order_id": order.order_id,
                "contract_id": self.symbol,
                "side": order.side,
                "order_type": "OPEN" if order.side == self.config.direction else "CLOSE",
                "status": order.status,
                "size": str(order.size),
                "price": str(order.price),
                "filled_size": str(order.filled_size),
            })

    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        if self._ws_task is None or self._ws_task.done():
            raise ConnectionError("Bulk market data is disconnected")
        if time.monotonic() - self._book_update_time > 30:
            raise TimeoutError("Bulk order book has not updated for 30 seconds")
        if contract_id != self.symbol or self._best_bid is None or self._best_ask is None:
            raise ValueError("Bulk order book has no bid or ask")
        if self._best_bid <= 0 or self._best_ask <= self._best_bid:
            raise ValueError("Bulk order book is invalid")
        return self._best_bid, self._best_ask

    async def get_order_price(self, direction: str) -> Decimal:
        bid, ask = await self.fetch_bbo_prices(self.symbol)
        price = ask - self.config.tick_size if direction == "buy" else bid + self.config.tick_size
        return self.round_to_tick(price)

    async def _post_signed(self, signed: Dict[str, Any]) -> Any:
        body = {key: signed[key] for key in ("actions", "nonce", "account", "signer", "signature")}
        if self.http is None:
            self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        async with self.http.post(f"{self.api_url}/order", json=body) as response:
            data = await response.json(content_type=None)
            if response.status >= 400 and not isinstance(data, dict):
                response.raise_for_status()
            return data

    @staticmethod
    def _ack_statuses(response: Any) -> List[Dict[str, Any]]:
        if not isinstance(response, dict):
            raise RuntimeError("Bulk order response must be an object")
        if response.get("status") == "error":
            raise RuntimeError(str(response.get("message") or "Bulk rejected transaction"))
        direct = response.get("response")
        if isinstance(direct, dict):
            statuses = direct.get("data", {}).get("statuses", [])
        else:
            payload = response.get("payload", response.get("data", {}))
            if isinstance(payload, dict) and "payload" in payload:
                payload = payload["payload"]
            nested = payload.get("response", {}) if isinstance(payload, dict) else {}
            statuses = nested.get("data", {}).get("statuses", []) if isinstance(nested, dict) else []
        if not isinstance(statuses, list) or not statuses:
            raise RuntimeError("Bulk acknowledgement lacks an order status")
        return statuses

    @staticmethod
    def _one_status(response: Any) -> Tuple[str, Dict[str, Any]]:
        statuses = BulkClient._ack_statuses(response)
        if len(statuses) != 1 or not isinstance(statuses[0], dict) or len(statuses[0]) != 1:
            raise RuntimeError("Bulk acknowledgement lacks exactly one order status")
        name, body = next(iter(statuses[0].items()))
        return str(name), body if isinstance(body, dict) else {}

    async def _place_limit(self, quantity: Decimal, price: Decimal, side: str, reduce_only: bool) -> OrderResult:
        if self._ws_task is None or self._ws_task.done():
            return OrderResult(success=False, error_message="Bulk trading connection is unavailable")
        if quantity <= 0 or quantity % self._lot_size != 0 or price <= 0:
            return OrderResult(success=False, error_message="Invalid Bulk price or order quantity")
        if not reduce_only and quantity * price < self._min_notional:
            raise ValueError(f"Bulk minimum notional is {self._min_notional} USD")
        signed = self._signer.sign({
            "type": "order",
            "symbol": self.symbol,
            "is_buy": side == "buy",
            "price": float(price),
            "size": float(quantity),
            "reduce_only": reduce_only,
            "order_type": {"type": "limit", "tif": "ALO"},
        })
        order_id = str(signed.get("order_id") or "")
        if not order_id:
            raise RuntimeError("Bulk signer did not compute an order id")
        try:
            status_name, body = self._one_status(await self._post_signed(signed))
            acknowledged = str(body.get("oid") or "")
            if acknowledged and acknowledged != order_id:
                raise RuntimeError("Bulk acknowledgement order id does not match signed order")
            normalized = status_name.lower().replace("_", "").replace("-", "")
            if normalized == "error" or normalized.startswith("rejected"):
                return OrderResult(False, error_message=str(body.get("message") or status_name))
            if normalized.startswith("cancel"):
                return OrderResult(False, error_message=str(body.get("message") or status_name))
            status = self._status(status_name)
            self._remember(OrderInfo(
                order_id, side, quantity, price, status,
                filled_size=self._fill_totals.get(order_id, Decimal(0)),
                remaining_size=quantity,
            ))
            stored = self._orders[order_id]
            return OrderResult(True, order_id, side, quantity, price, stored.status,
                               filled_size=stored.filled_size)
        except Exception as exc:
            # A timed-out post may have reached the exchange. Do not silently retry it.
            raise RuntimeError(f"Bulk order outcome is unknown: {exc}") from exc

    async def place_open_order(self, contract_id: str, quantity: Decimal, direction: str) -> OrderResult:
        price = await self.get_order_price(direction)
        return await self._place_limit(quantity, price, direction, reduce_only=False)

    async def place_close_order(self, contract_id: str, quantity: Decimal, price: Decimal, side: str) -> OrderResult:
        bid, ask = await self.fetch_bbo_prices(contract_id)
        if side == "sell":
            price = max(price, bid + self.config.tick_size)
        else:
            price = min(price, ask - self.config.tick_size)
        return await self._place_limit(quantity, self.round_to_tick(price), side, reduce_only=True)

    async def cancel_order(self, order_id: str) -> OrderResult:
        order = await self.get_order_info(order_id)
        if order is None:
            raise RuntimeError(f"Bulk order {order_id} cannot be located for cancellation")
        if order.status in {"FILLED", "CANCELED"}:
            return OrderResult(True, order_id, filled_size=order.filled_size)
        if self._ws_task is None or self._ws_task.done():
            raise ConnectionError("Bulk trading connection is unavailable")
        try:
            signed = self._signer.sign({
                "type": "cancel",
                "symbol": self.symbol,
                "order_id": order_id,
            })
            status_name, body = self._one_status(await self._post_signed(signed))
            normalized = status_name.lower().replace("_", "").replace("-", "")
            if normalized == "error" or normalized.startswith("rejected"):
                raise RuntimeError(str(body.get("message") or status_name))
            event = self._order_events.setdefault(order_id, asyncio.Event())
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            latest = self._orders.get(order_id, order)
            if latest.status not in {"FILLED", "CANCELED"}:
                latest.status = "CANCELED"
                self._remember(latest)
            stored = self._orders[order_id]
            return OrderResult(True, order_id, filled_size=stored.filled_size)
        except Exception as exc:
            raise RuntimeError(f"Bulk cancel outcome is unknown: {exc}") from exc

    @staticmethod
    def _parse_order(raw: Dict[str, Any]) -> OrderInfo:
        raw = raw.get("openOrder", raw)
        order_id = raw.get("oid", raw.get("orderId"))
        symbol = raw.get("sym", raw.get("coin", raw.get("symbol")))
        if not order_id or not symbol:
            raise ValueError("Bulk order is missing id or symbol")
        signed_size = Decimal(str(raw.get("sz", raw.get("size", 0))))
        filled = Decimal(str(raw.get("fillSz", raw.get("filledSz", 0))))
        original = Decimal(str(raw.get("origSz", abs(signed_size) + filled)))
        side = BulkClient._side_name(raw.get("isBuy", raw.get("side")), signed_size)
        size = abs(original)
        return OrderInfo(str(order_id), side, size,
                         Decimal(str(raw.get("px", raw.get("price", 0)))),
                         BulkClient._status(raw.get("status", "OPEN")),
                         filled_size=filled,
                         remaining_size=max(size - filled, Decimal(0)))

    async def get_active_orders(self, contract_id: str) -> List[OrderInfo]:
        account = await self._account()
        orders = []
        for raw in account["openOrders"]:
            order = self._parse_order(raw)
            payload = raw.get("openOrder", raw)
            symbol = payload.get("sym", payload.get("coin", payload.get("symbol")))
            if symbol == contract_id:
                self._orders[order.order_id] = order
                orders.append(order)
        return orders

    async def get_order_info(self, order_id: str) -> Optional[OrderInfo]:
        cached = self._orders.get(order_id)
        if cached is not None and cached.status in {"FILLED", "CANCELED"}:
            return cached
        for order in await self.get_active_orders(self.symbol):
            if order.order_id == order_id:
                return order
        cached = self._orders.get(order_id)
        if cached is not None and cached.status in {"FILLED", "CANCELED"}:
            return cached
        raise RuntimeError(f"Bulk order {order_id} is absent without a confirmed final state")

    async def get_account_positions(self) -> Decimal:
        account = await self._account()
        for position in account["positions"]:
            if position.get("symbol") == self.symbol:
                return Decimal(str(position["size"]))
        return Decimal(0)
