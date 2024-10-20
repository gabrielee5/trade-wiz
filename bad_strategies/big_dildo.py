import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
import matplotlib.pyplot as plt
from data_file_saving import save_trade_data_to_csv, convert_csv_to_xlsx
import talib as ta

# looks shit 

def calculate_volume_spike(volume, period=5, multiplier=5):
    avg_volume = volume.rolling(window=period).mean().shift(1)
    volume_spike = volume > (avg_volume * multiplier)
    return volume_spike

def preprocess_data(df):
    df = df.rename(columns={col: col.capitalize() for col in df.columns})
    df['VolumeSpikeSignal'] = calculate_volume_spike(df['Volume'])
    df['IsGreenCandle'] = df['Close'] > df['Open']
    return df

class VolumeSpikeStrategy(Strategy):
    def init(self):
        self.volume_spike = self.I(lambda: self.data.VolumeSpikeSignal)
        self.is_green_candle = self.I(lambda: self.data.IsGreenCandle)
        self.spike_high = None
        self.spike_low = None
        self.possible_long = False
        self.possible_short = False

    def next(self):
        if self.volume_spike[-1] and self.is_green_candle[-1]:
            self.possible_long = True
            self.possible_short = False
            self.spike_high = self.data.High[-1]
            self.spike_low = self.data.Low[-1]

        if self.volume_spike[-1] and not self.is_green_candle[-1]:
            self.possible_short = True
            self.possible_long = False
            self.spike_high = self.data.High[-1]
            self.spike_low = self.data.Low[-1]

        if self.possible_long: # Bullish signal
            if not self.position:
                if self.data.Close[-1] > self.spike_high:
                    self.buy(sl=self.spike_low)
            elif self.position.is_short:
                if self.data.Close[-1] > self.spike_high:
                    self.position.close()
                    self.buy(sl=self.spike_low)

        elif self.possible_short:  # Bearish signal
            if not self.position:
                if self.data.Close[-1] < self.spike_low:
                    self.sell(sl=self.spike_high)
            elif self.position.is_long:
                if self.data.Close[-1] < self.spike_low:
                    self.position.close()
                    self.sell(sl=self.spike_high)

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
    df = data.copy()
    df['Buy_Signal'] = 0
    df['Sell_Signal'] = 0
    df['Exit_Signal'] = 0

    for trade in results['_trades'].itertuples():
        if trade.Size > 0:  # Buy trade
            df.loc[trade.EntryTime, 'Buy_Signal'] = 1
            df.loc[trade.ExitTime, 'Exit_Signal'] = 1
        elif trade.Size < 0:  # Sell trade
            df.loc[trade.EntryTime, 'Sell_Signal'] = 1
            df.loc[trade.ExitTime, 'Exit_Signal'] = 1

    columns_to_display = ['Open', 'High', 'Low', 'Close', 'Volume', 
                          'VolumeSpikeSignal', 'IsGreenCandle',
                          'Buy_Signal', 'Sell_Signal', 'Exit_Signal']
    
    df_display = df[columns_to_display]

    trade_indices = df_display.index[
        (df_display['Buy_Signal'] == 1) | 
        (df_display['Sell_Signal'] == 1) | 
        (df_display['Exit_Signal'] == 1)
    ]

    def get_trade_rows(idx):
        start_idx = max(0, df_display.index.get_loc(idx) - 6)
        return df_display.iloc[start_idx:df_display.index.get_loc(idx) + 1]

    trade_rows = pd.concat([get_trade_rows(idx) for idx in trade_indices])
    trade_rows = trade_rows.drop_duplicates()

    return trade_rows

def display_dataframe_with_trades(trade_rows):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\nDataframe with Trade Signals (including two previous rows for each trade):")
    print(trade_rows.to_string())

def main():
    data = load_data('data/BTCUSDT_perp_1h_concatenated.csv')
    data = preprocess_data(data)
    # print(data.tail(20))
    bt = Backtest(data, VolumeSpikeStrategy, cash=100000, commission=.002)
    results = bt.run()
    
    print(results)
    
    # Uncomment the following lines if you want to see the trades table and plot
    # print_trades_table(results)
    custom_plot(results, data)

    # trade_data = get_trade_data(data, results)
    # display_dataframe_with_trades(trade_data)
    
    # file_path = save_trade_data_to_csv(trade_data, strategy_name="VolumeSpikeStrategy", asset_name="ETHUSDT")
    # convert_csv_to_xlsx(file_path)

if __name__ == "__main__":
    main()