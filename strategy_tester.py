import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
import mplfinance as mpf
import os
from data_file_saving import save_trade_data_to_csv, convert_csv_to_xlsx

# template to test new strategies
# the interesting strategies will be saves in a separate folder

def calculate_atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
    atr = tr.rolling(window=period).mean()
    return atr

def resample_to_interval(df, interval):
    # Ensure column names are capitalized
    df = df.rename(columns={col: col.capitalize() for col in df.columns})
    
    df_resampled = df.resample(f'{interval}min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    })
    return df_resampled

def preprocess_data(df, resample_interval):
    # Ensure column names are capitalized
    df = df.rename(columns={col: col.capitalize() for col in df.columns})
    
    # Resample data
    resampled = resample_to_interval(df, resample_interval)
    
    # Calculate indicators on resampled data
    resampled['ATR'] = calculate_atr(resampled.High, resampled.Low, resampled.Close) * 2.0
    resampled['CloseChange'] = resampled.Close.diff()
    resampled['AbsCloseChange'] = resampled['CloseChange'].abs()
    
    # Merge resampled data back to original timeframe
    for col in resampled.columns:
        df[f'Resample_{resample_interval}_{col}'] = resampled[col].reindex(df.index).ffill()
    
    return df

class ATRtrend(Strategy):
    resample_int = 60 * 3  # 3 hours
    
    def init(self):
        # Use pre-calculated indicators
        self.atr = self.I(lambda: self.data.df[f'Resample_{self.resample_int}_ATR'])
        self.close_change = self.I(lambda: self.data.df[f'Resample_{self.resample_int}_CloseChange'])
        self.abs_close_change = self.I(lambda: self.data.df[f'Resample_{self.resample_int}_AbsCloseChange'])


    def next(self):
        if self.close_change[-1] > self.atr[-1]:
            if not self.position:
                self.buy()
            elif self.position.is_short:
                self.position.close()
                self.buy() # inverts the position
        
        elif self.close_change[-1] * -1 > self.atr[-1]:
            if not self.position:
                self.sell()
            elif self.position.is_long:
                self.position.close()
                self.sell() # inverts the position

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    df = df.rename(columns={col: col.capitalize() for col in df.columns})
    return df

def print_trades_table(results):
    trades = results['_trades']
    print("\nTrades Table:")
    print(trades.to_string(index=False))
    
def custom_plot(results, original_data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot equity curve
    equity_curve = results['_equity_curve']
    equity_curve['Equity'].plot(ax=ax1)
    ax1.set_ylabel('Equity')
    ax1.set_title('Backtest Results')
    
    # Plot price
    original_data['Close'].plot(ax=ax2)
    ax2.set_ylabel('Price')
    
    # Plot buy and sell signals
    for trade in results['_trades'].itertuples():
        if trade.Size > 0:  # Buy signal
            ax2.scatter(trade.EntryTime, trade.EntryPrice, marker='^', color='g', s=100)
            ax2.scatter(trade.ExitTime, trade.ExitPrice, marker='o', color='g', s=100)
        elif trade.Size < 0:  # Sell signal
            ax2.scatter(trade.EntryTime, trade.EntryPrice, marker='v', color='r', s=100)
            ax2.scatter(trade.ExitTime, trade.ExitPrice, marker='o', color='r', s=100)
    
    plt.tight_layout()
    plt.show()

def get_trade_data(data, results):
    # Create a copy of the dataframe to avoid modifying the original
    df = data.copy()

    # Add columns for trade signals
    df['Buy_Signal'] = 0
    df['Sell_Signal'] = 0
    df['Exit_Signal'] = 0

    # Mark the trade signals
    for trade in results['_trades'].itertuples():
        if trade.Size > 0:  # Buy trade
            df.loc[trade.EntryTime, 'Buy_Signal'] = 1
            df.loc[trade.ExitTime, 'Exit_Signal'] = 1
        elif trade.Size < 0:  # Sell trade
            df.loc[trade.EntryTime, 'Sell_Signal'] = 1
            df.loc[trade.ExitTime, 'Exit_Signal'] = 1

    # Select relevant columns
    columns_to_display = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          f'Resample_{ATRtrend.resample_int}_ATR', 
                          f'Resample_{ATRtrend.resample_int}_CloseChange',
                          'Buy_Signal', 'Sell_Signal', 'Exit_Signal']
    
    df_display = df[columns_to_display]

    # Get indices of rows with trade signals
    trade_indices = df_display.index[
        (df_display['Buy_Signal'] == 1) | 
        (df_display['Sell_Signal'] == 1) | 
        (df_display['Exit_Signal'] == 1)
    ]

    # Function to get rows around a trade
    def get_trade_rows(idx):
        start_idx = max(0, df_display.index.get_loc(idx) - 6)
        return df_display.iloc[start_idx:df_display.index.get_loc(idx) + 1]

    # Collect rows for all trades
    trade_rows = pd.concat([get_trade_rows(idx) for idx in trade_indices])

    # Remove duplicate rows (in case of consecutive trades)
    trade_rows = trade_rows.drop_duplicates()

    return trade_rows

def display_dataframe_with_trades(trade_rows):
    # Display the dataframe
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\nDataframe with Trade Signals (including two previous rows for each trade):")
    print(trade_rows.to_string())

def main():
    # Load data
    data = load_data('data/ETHUSDT_perp_1h_concatenated.csv')

    # Preprocess data
    data = preprocess_data(data, ATRtrend.resample_int)

    # Create a Backtest instance
    bt = Backtest(data, ATRtrend, cash=100000, commission=.002)
    
    # Run the backtest
    results = bt.run()
    
    # Print results summary
    print(results)
    
    # Print trades table
    # print_trades_table(results)
    
    # Plot the backtest results using custom function
    # custom_plot(results, data)

    # Get and display trade data
    trade_data = get_trade_data(data, results)
    display_dataframe_with_trades(trade_data)
    
    # Save trade data to CSV
    file_path = save_trade_data_to_csv(trade_data, strategy_name="ATRtrend", asset_name="ETHUSDT")
    convert_csv_to_xlsx(file_path)

if __name__ == "__main__":
    main()
