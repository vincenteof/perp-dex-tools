"""BULK perpetual exchange adapter for the single-exchange trading bot."""

import asyncio
import os
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from bulk_api import BulkWebSocketClient
from bulk_api.common import Side, TimeInForce, Topic
from bulk_api.common.signer import TransactionSigner

from .base import BaseExchangeClient, OrderInfo, OrderResult


class BulkClient(BaseExchangeClient):
    API_URL = "https://exchange-api.bulk.trade/api/v1"
    WS_URL = "wss://exchange-ws1.bulk.trade"

    def __init__(self, config):
        super().__init__(config)
        self.symbol = f"{config.ticker.upper()}-USD"
        self.signer = TransactionSigner(os.environ["BULK_PRIVATE_KEY"])
        self.api_url = os.getenv("BULK_API_URL", self.API_URL).rstrip("/")
        self.ws_url = os.getenv("BULK_WS_URL", self.WS_URL)
        self.http: Optional[aiohttp.ClientSession] = None
        self.ws: Optional[BulkWebSocketClient] = None
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

    def setup_order_update_handler(self, handler) -> None:
        self._handler = handler

    async def _request(self, method: str, path: str, **kwargs) -> Any:
        if self.http is None:
            self.http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        async with self.http.request(method, f"{self.api_url}/{path}", **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def _account(self) -> Dict[str, Any]:
        data = await self._request("POST", "account", json={
            "type": "fullAccount", "user": self.signer.public_key,
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
        self.ws = BulkWebSocketClient(url=self.ws_url, symbols=[self.symbol], signer=self.signer)
        self.ws.on(Topic.ORDER, self._on_order)
        self.ws.on(Topic.FILL, self._on_fill)
        self.ws.on(Topic.L2SNAPSHOT, self._on_book_update)
        self.ws.on(Topic.L2DELTA, self._on_book_update)
        if not await self.ws.connect():
            raise ConnectionError("Failed to connect to Bulk WebSocket")
        await self.ws.subscribe_orderbook_snapshot(self.symbol, nlevels=20)
        await self.ws.subscribe_orderbook_delta(self.symbol)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            book = self.ws.get_book(self.symbol)
            if (self.ws.account_snapshot is not None and self._book_update_time
                    and book and book.get_best_bid() and book.get_best_ask()):
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("Bulk account or order book snapshot did not arrive")

    async def disconnect(self) -> None:
        try:
            if self.ws is not None:
                await self.ws.disconnect()
        finally:
            if self.http is not None:
                await self.http.close()
                self.http = None

    @staticmethod
    def _status(status: Any) -> str:
        name = getattr(status, "name", str(status)).upper()
        if name in {"RESTING", "WORKING", "CANCEL_PENDING", "PLACED"}:
            return "OPEN"
        if name == "PARTIALLYFILLED":
            return "PARTIALLY_FILLED"
        if name.startswith("CANCELLED") or name.startswith("CANCELED"):
            return "CANCELED"
        return name

    @staticmethod
    def _side(side: Any) -> str:
        if side == Side.BUY or str(side).upper() in {"BUY", "BID", "TRUE"}:
            return "buy"
        if side == Side.SELL or str(side).upper() in {"SELL", "ASK", "FALSE"}:
            return "sell"
        raise ValueError(f"Unknown Bulk order side: {side}")

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

    def _on_order(self, state) -> None:
        if state.symbol != self.symbol or not state.order_id:
            return
        prior = self._orders.get(state.order_id)
        filled = max(Decimal(str(state.size_done or 0)), self._fill_totals.get(state.order_id, Decimal(0)))
        size = Decimal(str(state.size_orig or 0))
        if size <= 0:
            size = (prior.size if prior else Decimal(str(state.size or 0)) + filled)
        price = Decimal(str(state.price or (prior.price if prior else 0)))
        order = OrderInfo(
            order_id=state.order_id,
            side=self._side(state.side),
            size=size,
            price=price,
            status=self._status(state.status),
            filled_size=filled,
            remaining_size=max(size - filled, Decimal(0)),
        )
        self._orders[order.order_id] = order
        self._emit(order)
        if order.status in {"FILLED", "CANCELED"}:
            self._order_events.setdefault(order.order_id, asyncio.Event()).set()

    def _on_fill(self, fill) -> None:
        if fill.symbol != self.symbol or not fill.order_id:
            return
        total = self._fill_totals.get(fill.order_id, Decimal(0)) + Decimal(str(fill.size))
        self._fill_totals[fill.order_id] = total
        order = self._orders.get(fill.order_id)
        if order is None:
            return  # The order update or placement response will establish the order.
        order.filled_size = max(order.filled_size, total)
        order.remaining_size = max(order.size - order.filled_size, Decimal(0))
        if order.remaining_size == 0:
            order.status = "FILLED"
            self._order_events.setdefault(order.order_id, asyncio.Event()).set()
        elif order.status != "CANCELED":
            order.status = "PARTIALLY_FILLED"
        self._emit(order)

    def _on_book_update(self, update) -> None:
        if update.symbol == self.symbol:
            self._book_update_time = time.monotonic()

    async def fetch_bbo_prices(self, contract_id: str) -> Tuple[Decimal, Decimal]:
        if self.ws is None or not self.ws.is_connected:
            raise ConnectionError("Bulk market data is disconnected")
        if time.monotonic() - self._book_update_time > 30:
            raise TimeoutError("Bulk order book has not updated for 30 seconds")
        book = self.ws.get_book(contract_id)
        bid = book.get_best_bid() if book else None
        ask = book.get_best_ask() if book else None
        if not bid or not ask:
            raise ValueError("Bulk order book has no bid or ask")
        best_bid, best_ask = Decimal(str(bid.price)), Decimal(str(ask.price))
        if best_bid <= 0 or best_ask <= best_bid:
            raise ValueError("Bulk order book is invalid")
        return best_bid, best_ask

    async def get_order_price(self, direction: str) -> Decimal:
        bid, ask = await self.fetch_bbo_prices(self.symbol)
        price = ask - self.config.tick_size if direction == "buy" else bid + self.config.tick_size
        return self.round_to_tick(price)

    async def _place_limit(self, quantity: Decimal, price: Decimal, side: str, reduce_only: bool) -> OrderResult:
        if self.ws is None or not self.ws.is_connected:
            return OrderResult(success=False, error_message="Bulk trading connection is unavailable")
        if quantity <= 0 or quantity % self._lot_size != 0 or price <= 0:
            return OrderResult(success=False, error_message="Invalid Bulk price or order quantity")
        if not reduce_only and quantity * price < self._min_notional:
            raise ValueError(f"Bulk minimum notional is {self._min_notional} USD")
        try:
            response = await self.ws.place_limit_order(
                self.symbol,
                Side.BUY if side == "buy" else Side.SELL,
                float(price),
                float(quantity),
                reduce_only=reduce_only,
                time_in_force=TimeInForce.ALO,
            )
            if response.is_error():
                return OrderResult(success=False, error_message=response.message or str(response.status))
            if not response.order_id:
                raise RuntimeError("Bulk accepted an order without returning its id")
            status = self._status(response.status)
            existing = self._orders.get(response.order_id)
            if existing is None:
                filled = self._fill_totals.get(response.order_id, Decimal(0))
                self._orders[response.order_id] = OrderInfo(
                    response.order_id, side, quantity, price, status,
                    filled_size=filled, remaining_size=max(quantity - filled, Decimal(0)),
                )
                if filled >= quantity:
                    self._orders[response.order_id].status = "FILLED"
                    self._emit(self._orders[response.order_id])
                existing = self._orders[response.order_id]
            return OrderResult(True, response.order_id, side, quantity, price, existing.status,
                               filled_size=existing.filled_size)
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
        if self.ws is None or not self.ws.is_connected:
            raise ConnectionError("Bulk trading connection is unavailable")
        try:
            response = await self.ws.cancel_order(
                side=Side.BUY if order.side == "buy" else Side.SELL,
                symbol=self.symbol,
                order_id=order_id,
            )
            if response.is_error():
                raise RuntimeError(response.message or f"Bulk cancel rejected: {response.status}")
            event = self._order_events.setdefault(order_id, asyncio.Event())
            try:
                await asyncio.wait_for(event.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
            latest = await self.get_order_info(order_id)
            if latest is None or latest.status not in {"FILLED", "CANCELED"}:
                raise RuntimeError(f"Bulk cancel of {order_id} is unconfirmed")
            return OrderResult(True, order_id, filled_size=latest.filled_size)
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
        side = "buy" if raw.get("isBuy", signed_size >= 0) else "sell"
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
        # An order missing from the live book is only safe to classify when an
        # authenticated account event has supplied its terminal state.
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
