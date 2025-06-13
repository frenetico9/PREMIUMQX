import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import talib
import logging
import csv
import os
from queue import Queue
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Binance public API
exchange = ccxt.binance({'enableRateLimit': True})

# Crypto pairs
PAIRS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT', 'DOGE/USDT']

# Strategy parameters
TIMEFRAME = '15m'
ATR_PERIOD = 14
RISK_REWARD = 2.0
LOOKBACK = 50
KILL_ZONE_HOURS = [(8, 12), (14, 18)]  # London, NY sessions (UTC)
BACKTEST_DAYS = 60

# Fetch historical data
def fetch_data(symbol, timeframe, since):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD)
        df['sma20'] = talib.SMA(df['close'], timeperiod=20)
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

# Check kill zone
def in_kill_zone(timestamp):
    hour = timestamp.hour
    return any(start <= hour < end for start, end in KILL_ZONE_HOURS)

# Market structure analysis
def get_market_structure(df, lookback):
    highs = df['high'].rolling(window=lookback).max()
    lows = df['low'].rolling(window=lookback).min()
    current_high, current_low = df['high'].iloc[-1], df['low'].iloc[-1]
    prev_high, prev_low = highs.iloc[-2], lows.iloc[-2]
    
    if current_high > prev_high and current_low > prev_low:
        return 'bullish'
    elif current_high < prev_high and current_low < prev_low:
        return 'bearish'
    return 'ranging'

# Find order block and fair value gap
def find_order_block(df, structure):
    for i in range(len(df)-2, len(df)-10, -1):
        if structure == 'bullish' and df['close'].iloc[i] < df['open'].iloc[i]:
            return df['high'].iloc[i], df['low'].iloc[i]
        elif structure == 'bearish' and df['close'].iloc[i] > df['open'].iloc[i]:
            return df['high'].iloc[i], df['low'].iloc[i]
    return None, None

# Generate trading signal
def generate_signal(df, symbol):
    try:
        structure = get_market_structure(df, LOOKBACK)
        ob_high, ob_low = find_order_block(df, structure)
        current_price = df['close'].iloc[-1]
        atr = df['atr'].iloc[-1]
        timestamp = df['timestamp'].iloc[-1]
        sma20 = df['sma20'].iloc[-1]

        if not in_kill_zone(timestamp) or ob_high is None or ob_low is None:
            return None, None, None

        signal = None
        sl = None
        tp = None

        if structure == 'bullish' and current_price > ob_high and current_price > sma20:
            signal = 'buy'
            sl = ob_low - 1.5 * atr
            tp = current_price + RISK_REWARD * (current_price - sl)
        elif structure == 'bearish' and current_price < ob_low and current_price < sma20:
            signal = 'sell'
            sl = ob_high + 1.5 * atr
            tp = current_price - RISK_REWARD * (sl - current_price)

        return signal, sl, tp
    except Exception as e:
        logger.error(f"Error generating signal for {symbol}: {e}")
        return None, None, None

# Backtest strategy
def backtest(symbols, days, queue):
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    results = []
    
    for symbol in symbols:
        try:
            df = fetch_data(symbol, TIMEFRAME, since)
            if df is None or len(df) < LOOKBACK:
                continue

            trades = []
            equity = [100000]  # Starting balance: 100k USDT
            for i in range(LOOKBACK, len(df)):
                signal, sl, tp = generate_signal(df.iloc[:i+1], symbol)
                if signal:
                    entry_price = df['close'].iloc[i]
                    for j in range(i+1, len(df)):
                        high, low = df['high'].iloc[j], df['low'].iloc[j]
                        if signal == 'buy':
                            if high >= tp:
                                profit = (tp - entry_price) / entry_price * 100
                                trades.append(profit)
                                equity.append(equity[-1] * (1 + profit/100))
                                break
                            elif low <= sl:
                                loss = (sl - entry_price) / entry_price * 100
                                trades.append(loss)
                                equity.append(equity[-1] * (1 + loss/100))
                                break
                        elif signal == 'sell':
                            if low <= tp:
                                profit = (entry_price - tp) / entry_price * 100
                                trades.append(profit)
                                equity.append(equity[-1] * (1 + profit/100))
                                break
                            elif high >= sl:
                                loss = (entry_price - sl) / entry_price * 100
                                trades.append(loss)
                                equity.append(equity[-1] * (1 + loss/100))
                                break
            
            win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
            total_profit = sum(trades)
            drawdown = max(0, (max(equity) - min(equity)) / max(equity) * 100) if equity else 0
            profit_factor = sum([t for t in trades if t > 0]) / abs(sum([t for t in trades if t < 0])) if any(t < 0 for t in trades) else float('inf')
            sharpe_ratio = np.mean(trades) / np.std(trades) * np.sqrt(365 * 24 * 4) if trades else 0

            results.append({
                'symbol': symbol,
                'trades': len(trades),
                'win_rate': win_rate,
                'profit': total_profit,
                'drawdown': drawdown,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio
            })
        except Exception as e:
            logger.error(f"Error in backtest for {symbol}: {e}")
    
    queue.put(results)

# Export backtest results to CSV
def export_to_csv(results):
    try:
        filename = f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['symbol', 'trades', 'win_rate', 'profit', 'drawdown', 'profit_factor', 'sharpe_ratio'])
            writer.writeheader()
            writer.writerows(results)
        return filename
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return None

# GUI
class SignalBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Crypto Signal Bot")
        self.style = ttkb.Style(theme='darkly')
        self.root.geometry("1000x600")
        self.running = False
        self.queue = Queue()

        # Tabs
        self.notebook = ttkb.Notebook(self.root)
        self.notebook.pack(pady=10, padx=10, fill='both', expand=True)
        
        self.signal_frame = ttkb.Frame(self.notebook)
        self.backtest_frame = ttkb.Frame(self.notebook)
        self.notebook.add(self.signal_frame, text="Live Signals")
        self.notebook.add(self.backtest_frame, text="Backtest Results")

        # Signal Frame
        self.signal_text = scrolledtext.ScrolledText(self.signal_frame, wrap=tk.WORD, width=80, height=20, font=('Arial', 10))
        self.signal_text.pack(padx=10, pady=10)
        self.monitor_button = ttkb.Button(self.signal_frame, text="Start Monitoring", bootstyle=SUCCESS, command=self.start_monitoring)
        self.monitor_button.pack(pady=5)

        # Backtest Frame
        self.backtest_button = ttkb.Button(self.backtest_frame, text="Run 60-Day Backtest", bootstyle=PRIMARY, command=self.run_backtest)
        self.backtest_button.pack(pady=5)
        
        # Results Table
        columns = ('Symbol', 'Trades', 'Win Rate (%)', 'Profit (%)', 'Drawdown (%)', 'Profit Factor', 'Sharpe Ratio')
        self.tree = ttkb.Treeview(self.backtest_frame, columns=columns, show='headings', bootstyle=INFO)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor='center')
        self.tree.pack(padx=10, pady=10, fill='both', expand=True)
        
        self.export_button = ttkb.Button(self.backtest_frame, text="Export to CSV", bootstyle=SECONDARY, command=self.export_results)
        self.export_button.pack(pady=5)
        self.status_label = ttkb.Label(self.backtest_frame, text="", bootstyle=INFO)
        self.status_label.pack(pady=5)

    def run_backtest(self):
        self.backtest_button.config(state='disabled')
        self.status_label.config(text="Running backtest...")
        threading.Thread(target=backtest, args=(PAIRS, BACKTEST_DAYS, self.queue), daemon=True).start()
        self.root.after(100, self.check_backtest_results)

    def check_backtest_results(self):
        if not self.queue.empty():
            results = self.queue.get()
            for item in self.tree.get_children():
                self.tree.delete(item)
            for result in results:
                self.tree.insert('', 'end', values=(
                    result['symbol'],
                    result['trades'],
                    f"{result['win_rate']:.2f}",
                    f"{result['profit']:.2f}",
                    f"{result['drawdown']:.2f}",
                    f"{result['profit_factor']:.2f}",
                    f"{result['sharpe_ratio']:.2f}"
                ))
            self.backtest_button.config(state='normal')
            self.status_label.config(text="Backtest complete!")
            self.results = results
        else:
            self.root.after(100, self.check_backtest_results)

    def export_results(self):
        if hasattr(self, 'results'):
            filename = export_to_csv(self.results)
            if filename:
                self.status_label.config(text=f"Results exported to {filename}")
            else:
                self.status_label.config(text="Error exporting results")
        else:
            self.status_label.config(text="No results to export")

    def monitor_signals(self):
        while self.running:
            try:
                for symbol in PAIRS:
                    df = fetch_data(symbol, TIMEFRAME, since=int((datetime.now() - timedelta(days=1)).timestamp() * 1000))
                    if df is None:
                        continue
                    signal, sl, tp = generate_signal(df, symbol)
                    if signal:
                        msg = f"{datetime.now()}: {symbol} - {signal.upper()} at {df['close'].iloc[-1]:.2f}, SL: {sl:.2f}, TP: {tp:.2f}\n"
                        self.signal_text.insert(tk.END, msg)
                        self.signal_text.see(tk.END)
                self.root.update()
                import time
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")

    def start_monitoring(self):
        if not self.running:
            self.running = True
            self.monitor_button.config(text="Stop Monitoring", bootstyle=WARNING, command=self.stop_monitoring)
            threading.Thread(target=self.monitor_signals, daemon=True).start()
        else:
            self.stop_monitoring()

    def stop_monitoring(self):
        self.running = False
        self.monitor_button.config(text="Start Monitoring", bootstyle=SUCCESS, command=self.start_monitoring)

if __name__ == "__main__":
    root = ttkb.Window(themename="darkly")
    app = SignalBotGUI(root)
    root.mainloop()
