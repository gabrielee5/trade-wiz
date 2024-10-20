import ccxt.pro as ccxtpro
from asyncio import run
import os
from dotenv import load_dotenv
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the environment variables from .env file
load_dotenv()

async def main2():
    exchange = ccxtpro.bybit({'newUpdates': False})
    while True:
        orderbook = await exchange.watch_order_book('BTC/USDT:USDT')
        # print(orderbook['asks'][0], orderbook['bids'][0])
        print(orderbook)
    await exchange.close()

async def main():
    logging.info("Starting the balance watcher...")

    # Initialize the exchange with API keys from .env
    exchange = ccxtpro.bybit({
        'apiKey': os.getenv('BYBIT_API_KEY'),
        'secret': os.getenv('BYBIT_API_SECRET'),
        'enableRateLimit': True,
    })

    logging.info(f"Initialized {exchange.id} exchange")

    if exchange.has['watchBalance']:
        logging.info("Exchange supports watchBalance. Starting balance watch...")
        try:
            while True:
                try:
                    balance = await exchange.watch_balance()
                    logging.info(f"{exchange.iso8601(exchange.milliseconds())}: {balance}")
                except ccxtpro.NetworkError as e:
                    logging.error(f"Network error occurred: {e}")
                except ccxtpro.ExchangeError as e:
                    logging.error(f"Exchange error occurred: {e}")
                except Exception as e:
                    logging.error(f"An unexpected error occurred: {e}")
                    break  # Exit the loop on unexpected errors
                
                await asyncio.sleep(1)  # Add a small delay between requests
        except asyncio.CancelledError:
            logging.info("Balance watch cancelled")
        finally:
            await exchange.close()
            logging.info("Exchange connection closed")
    else:
        logging.error("This exchange does not support real-time balance updates.")

    logging.info("Balance watcher stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main2())
    except KeyboardInterrupt:
        logging.info("Script interrupted by user")
    except Exception as e:
        logging.error(f"An error occurred while running the script: {e}")