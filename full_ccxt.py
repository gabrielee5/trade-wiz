import os
from dotenv import load_dotenv
import ccxt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Exchange configuration
EXCHANGE = 'bybit'  # Change this to the exchange you want to use
SYMBOL = 'BTC/USDT'  # Change this to the trading pair you're interested in

def initialize_exchange():
    exchange_class = getattr(ccxt, EXCHANGE)
    exchange = exchange_class({
        'apiKey': os.getenv(f'{EXCHANGE.upper()}_API_KEY'),
        'secret': os.getenv(f'{EXCHANGE.upper()}_API_SECRET'),
        'enableRateLimit': True,
    })
    return exchange

def fetch_ticker(exchange):
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        logging.info(f"Ticker for {SYMBOL}: {ticker}")
        return ticker
    except ccxt.BaseError as e:
        logging.error(f"Error fetching ticker: {str(e)}")

def fetch_orderbook(exchange):
    try:
        orderbook = exchange.fetch_order_book(SYMBOL)
        logging.info(f"Orderbook for {SYMBOL}: Top 5 bids and asks")
        logging.info(f"Bids: {orderbook['bids'][:5]}")
        logging.info(f"Asks: {orderbook['asks'][:5]}")
        return orderbook
    except ccxt.BaseError as e:
        logging.error(f"Error fetching orderbook: {str(e)}")

def fetch_balance(exchange):
    try:
        balance = exchange.fetch_balance()
        logging.info(f"Balance: {balance['total']}")
        return balance
    except ccxt.BaseError as e:
        logging.error(f"Error fetching balance: {str(e)}")

def place_limit_order(exchange, side, amount, price):
    try:
        order = exchange.create_limit_order(SYMBOL, side, amount, price)
        logging.info(f"Placed {side} limit order: {order}")
        return order
    except ccxt.BaseError as e:
        logging.error(f"Error placing limit order: {str(e)}")

def main():
    exchange = initialize_exchange()
    
    fetch_ticker(exchange)
    fetch_orderbook(exchange)
    fetch_balance(exchange)
    
    # Uncomment the following line to test placing a limit order
    # place_limit_order(exchange, 'buy', 0.001, 50000)  # Buy 0.001 BTC at $50,000

if __name__ == "__main__":
    main()