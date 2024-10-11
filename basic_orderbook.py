# -*- coding: utf-8 -*-

import asyncio
import os
from dotenv import load_dotenv
import ccxt.async_support as ccxt  # noqa: E402

# Load environment variables from .env file
load_dotenv()

async def test():
    exchange = ccxt.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY'),
        'secret': os.getenv('BYBIT_API_SECRET'),
        'enableRateLimit': True,
    })

    try:
        orderbook = await exchange.fetch_order_book('BTC/USDT')
        await exchange.close()
        return orderbook
    except ccxt.BaseError as e:
        print(type(e).__name__, str(e), str(e.args))
        raise e

async def main():
    result = await test()
    print("Orderbook for BTC/USDT:")
    print(f"Bids: {result['bids'][:5]}")  # Print first 5 bids
    print(f"Asks: {result['asks'][:5]}")  # Print first 5 asks

asyncio.run(main())