#!/usr/bin/env python3
"""Public, read-only Arcus market + WebSocket check; no keys or orders."""

import argparse
import asyncio
import json

import aiohttp


async def probe(network, ticker):
    host = "api.arcus.xyz" if network == "mainnet" else "api.testnet.arcus.xyz"
    symbol = ticker.upper()
    if not symbol.endswith("-USD"):
        symbol += "-USD"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as http:
        async with http.get(f"https://{host}/v1/markets") as response:
            response.raise_for_status()
            data = await response.json()
    market = next(m for m in data["markets"] if m["marketDisplayName"] == symbol)
    print(json.dumps({k: market[k] for k in (
        "marketDisplayName", "marketId", "status", "tickSize", "stepSize",
        "minOrderNotional", "minOrderSize", "maxOrderSize")}, ensure_ascii=False))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as http:
        async with http.ws_connect(f"wss://{host}/v1/ws", heartbeat=15) as ws:
            await ws.send_json({"type": "subscribe", "channel": "l2Orderbook",
                                "id": symbol, "nLevels": 1})
            for _ in range(10):
                msg = await ws.receive(timeout=15)
                if msg.type != aiohttp.WSMsgType.TEXT:
                    raise ConnectionError(f"Arcus socket returned {msg.type.name}")
                frame = json.loads(msg.data)
                if frame.get("channel") == "l2Orderbook" and frame.get("contents", {}).get("bids"):
                    print("WebSocket snapshot:", json.dumps(frame["contents"]))
                    return
    raise ConnectionError("Arcus socket did not provide an order-book snapshot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", choices=["mainnet", "testnet"], default="testnet")
    parser.add_argument("--ticker", default="ETH")
    args = parser.parse_args()
    asyncio.run(asyncio.wait_for(probe(args.network, args.ticker), 45))
