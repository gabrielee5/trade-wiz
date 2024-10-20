import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt


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
        if self.close_change[-1] > self.atr[-2]:
            if not self.position:
                self.buy()
            elif self.position.is_short:
                self.position.close()
                self.buy() # inverts the position
        
        elif self.close_change[-1] * -1 > self.atr[-2]:
            if not self.position:
                self.sell()
            elif self.position.is_long:
                self.position.close()
                self.sell() # inverts the position

def load_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Convert timestamp from milliseconds to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Ensure column names are lowercase
    df.columns = df.columns.str.lower()
    
    # Ensure specific columns are present and in correct order
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in the data.")
    
    # Reorder columns to ensure they match the required format
    df = df[required_columns + [col for col in df.columns if col not in required_columns]]
    
    return df

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

def main():
    # Load data
    data = load_data('data/SOLUSDT_60_data.csv')

    # Preprocess data
    data = preprocess_data(data, ATRtrend.resample_int)

    # Create a Backtest instance
    bt = Backtest(data, ATRtrend, cash=100000, commission=.002)
    
    # Run the backtest
    results = bt.run()
    
    # Print results
    print(results)
    
    # Plot the backtest results using custom function
    custom_plot(results, data)

if __name__ == "__main__":
    main()
