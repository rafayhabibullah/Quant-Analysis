import requests
import os

# --- CONFIG ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

BINANCE_SYMBOL = "BTCUSDT"
INTERVAL = "1h"  # hourly candles

def fetch_last_two_closes():
    """
    Fetch last 2 hourly closing prices from Binance Klines API
    Returns: (previous_close, current_close)
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": BINANCE_SYMBOL,
        "interval": INTERVAL,
        "limit": 3  # last two candles
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    klines = response.json()
    print(response.json())
    if len(klines) < 3:
        raise ValueError("Not enough candle data from Binance")
    prev_close = float(klines[-3][4])  # 4 = close price
    curr_close = float(klines[-2][4])
    return prev_close, curr_close

def send_telegram(message):
    """Send message to Telegram chat using bot token."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    response = requests.post(url, data=payload)
    response.raise_for_status()

def main():
    try:
        old_price, new_price = fetch_last_two_closes()
        change = ((new_price - old_price) / old_price) * 100

        print(f"Previous: ${old_price:.2f}, Current: ${new_price:.2f}, Change: {change:.2f}%")

        direction = "📈 up" if change > 0 else "📉 down"
        message = f"Bitcoin moved {direction} {abs(change):.2f}%\n${old_price:.2f} → ${new_price:.2f}"
        send_telegram(message)
        print("Alert sent.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
