import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt

def calculate_atr(high, low, close, period=14):
    tr = np.maximum(high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
    atr = tr.rolling(window=period).mean()
    return atr

def hawkes_process(data: pd.Series, kappa: float):
    assert(kappa > 0.0)
    alpha = np.exp(-kappa)
    arr = data.to_numpy()
    output = np.zeros(len(data))
    output[:] = np.nan
    for i in range(1, len(data)):
        if np.isnan(output[i - 1]):
            output[i] = arr[i]
        else:
            output[i] = output[i - 1] * alpha + arr[i]
    return pd.Series(output, index=data.index) * kappa

def preprocess_data(df, kappa, lookback):
    # Ensure column names are capitalized
    df = df.rename(columns={col: col.capitalize() for col in df.columns})
    
    # Calculate normalized range
    df['ATR'] = calculate_atr(df.High, df.Low, df.Close)
    df['NormRange'] = (np.log(df.High) - np.log(df.Low)) / df['ATR']
    
    # Calculate Hawkes process
    df['VHawk'] = hawkes_process(df['NormRange'], kappa)
    
    # Calculate quantiles
    df['Q05'] = df['VHawk'].rolling(lookback).quantile(0.05)
    df['Q95'] = df['VHawk'].rolling(lookback).quantile(0.95)
    
    return df

class HawkesVolumeStrategy(Strategy):
    kappa = 0.1
    lookback = 168  # 7 days for hourly data
    
    def init(self):
        self.vhawk = self.I(lambda: self.data.VHawk)
        self.q05 = self.I(lambda: self.data.Q05)
        self.q95 = self.I(lambda: self.data.Q95)
        self.last_below = -1
        self.curr_sig = 0

    def next(self):
        if self.vhawk[-1] < self.q05[-1]:
            self.last_below = len(self.data) - 1
            self.curr_sig = 0
            if self.position:
                self.position.close()

        elif self.vhawk[-1] > self.q95[-1] and self.vhawk[-2] <= self.q95[-2] and self.last_below > 0:
            change = self.data.Close[-1] - self.data.Close[self.last_below]
            if change > 0.0:
                self.curr_sig = 1
                if not self.position.is_long:
                    if self.position:
                        self.position.close()
                    self.buy()
            else:
                self.curr_sig = -1
                if not self.position.is_short:
                    if self.position:
                        self.position.close()
                    self.sell()

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    df = df.rename(columns={col: col.capitalize() for col in df.columns})
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
    data = preprocess_data(data, HawkesVolumeStrategy.kappa, HawkesVolumeStrategy.lookback)

    # Create a Backtest instance
    bt = Backtest(data, HawkesVolumeStrategy, cash=100000, commission=.002)
    
    # Run the backtest
    results = bt.run()
    
    # Print results
    print(results)
    
    # Plot the backtest results using custom function
    custom_plot(results, data)

if __name__ == "__main__":
    main()