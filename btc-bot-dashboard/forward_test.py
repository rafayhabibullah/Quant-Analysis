"""
Live Forward Test — BTC/ETH 5m (Binance Testnet)
=================================================
Runs Consensus ≥4/5 strategy on real-time Binance market data.
Paper trading only — no real orders placed.

  Data  : testnet.binance.vision  (real prices, simulated account)
  Signal: Consensus ≥4/5 (best strategy from 500-day optimization)
  Entry : on candle CLOSE (no look-ahead)
  Risk  : 1.5% per trade  |  Fixed 2R target

Run: python forward_test.py [--capital 1000]
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import time
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from backtest_multi import precompute, combo_fires, build_trade
from optimize_directional import FIXED, BUY, SELL, _atr
from backtest_directional import WARMUP

# ─────────────────────────────────────────────────────────────── #
#  Config                                                          #
# ─────────────────────────────────────────────────────────────── #

TESTNET_BASE   = 'https://testnet.binance.vision/api/v3'
PROD_BASE      = 'https://api.binance.com/api/v3'
CANDLE_MINUTES = 5
MIN_VOTES      = 4
FETCH_LIMIT    = 300     # rolling window size (> WARMUP=115)
CANDLE_BUFFER  = 20      # seconds after candle close before reading signal
LOG_FILE       = 'forward_trades.csv'

# ─────────────────────────────────────────────────────────────── #
#  Data fetching                                                   #
# ─────────────────────────────────────────────────────────────── #

def fetch_candles(symbol: str, limit: int = FETCH_LIMIT) -> pd.DataFrame:
    """Fetch latest candles — tries testnet first, falls back to production."""
    import logging
    sym = symbol.replace('/', '')
    for base in [TESTNET_BASE, PROD_BASE]:
        try:
            r = requests.get(f'{base}/klines',
                             params={'symbol': sym,
                                     'interval': f'{CANDLE_MINUTES}m',
                                     'limit': limit},
                             timeout=15)
            r.raise_for_status()
            raw = r.json()
            df  = pd.DataFrame(raw, columns=[
                'ts','open','high','low','close','volume',
                'close_ts','qvol','trades','tb','tq','ignore'])
            df['datetime'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_localize(None)
            for col in ['open','high','low','close','volume']:
                df[col] = df[col].astype(float)
            return df[['datetime','open','high','low','close','volume']]
        except Exception as e:
            logging.warning(f'fetch_candles failed for {base}: {type(e).__name__}: {e}')
            continue
    raise RuntimeError('Both testnet and production Binance are unreachable')

def now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

def seconds_to_next_close() -> float:
    """Seconds until the next 5m candle closes + buffer."""
    now       = time.time()
    interval  = CANDLE_MINUTES * 60
    next_close= (now // interval + 1) * interval
    return max(1, next_close - now + CANDLE_BUFFER)

# ─────────────────────────────────────────────────────────────── #
#  Signal checker  (Consensus ≥4/5 on last candle)                #
# ─────────────────────────────────────────────────────────────── #

def check_signal(btc: pd.DataFrame, eth: pd.DataFrame):
    """
    Run Consensus ≥MIN_VOTES on the last candle of the rolling window.
    Returns (direction, combo) or (None, None).
    """
    df  = btc.reset_index(drop=True)
    sec = eth.reset_index(drop=True)
    idx = len(df) - 1

    if idx < WARMUP:
        return None, None

    # Session / time filter
    ts = df.iloc[idx]['datetime']
    h  = ts.hour
    if (0 <= h < 4) or h in FIXED['blocked_hours']:
        return None, None

    long_pre, short_pre, ema9, atr, vol_sma = precompute(df, sec)

    long_voters  = [p for p in long_pre  if combo_fires(p, df, ema9, atr, vol_sma, idx, 'long')]
    short_voters = [p for p in short_pre if combo_fires(p, df, ema9, atr, vol_sma, idx, 'short')]

    if len(long_voters) >= MIN_VOTES:
        c = dict(long_voters[0]); c['votes'] = len(long_voters)
        return 'long', c
    if len(short_voters) >= MIN_VOTES:
        c = dict(short_voters[0]); c['votes'] = len(short_voters)
        return 'short', c
    return None, None

# ─────────────────────────────────────────────────────────────── #
#  Trade exit checker                                              #
# ─────────────────────────────────────────────────────────────── #

def check_exit(active: dict, candle: pd.Series):
    """Returns (exit_price, pnl) if trade closed, else (None, None)."""
    hi, lo = candle['high'], candle['low']
    if active['sig'] == BUY:
        if lo <= active['sl']:
            return active['sl'], -active['risk_usd']
        if hi >= active['tgt']:
            return active['tgt'], active['risk_usd'] * FIXED['rr_ratio']
    else:
        if hi >= active['sl']:
            return active['sl'], -active['risk_usd']
        if lo <= active['tgt']:
            return active['tgt'], active['risk_usd'] * FIXED['rr_ratio']
    return None, None

# ─────────────────────────────────────────────────────────────── #
#  Trade logger                                                    #
# ─────────────────────────────────────────────────────────────── #

def log_trade(record: dict):
    df  = pd.DataFrame([record])
    hdr = not Path(LOG_FILE).exists()
    df.to_csv(LOG_FILE, mode='a', header=hdr, index=False)

# ─────────────────────────────────────────────────────────────── #
#  Display                                                         #
# ─────────────────────────────────────────────────────────────── #

G   = '\033[92m'
R   = '\033[91m'
Y   = '\033[93m'
B   = '\033[94m'
W   = '\033[97m'
DIM = '\033[2m'
RST = '\033[0m'

def print_status(capital, initial, active, trades, btc_price, next_in):
    os.system('clear')
    total_ret = (capital - initial) / initial * 100
    n         = len(trades)
    wins      = sum(1 for t in trades if t['pnl'] > 0)
    wr        = wins / n * 100 if n else 0
    gp        = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gl        = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    pf        = gp / gl if gl > 0 else float('inf')

    print(f"\n{'═'*58}")
    print(f"  {B}Forward Test — Consensus ≥4/5  |  BTC/ETH 5m{RST}")
    print(f"{'═'*58}")
    print(f"  BTC Price  : {W}${btc_price:>10,.2f}{RST}   "
          f"UTC {DIM}{now_utc().strftime('%Y-%m-%d %H:%M:%S')}{RST}")
    print(f"  Next check : {DIM}in ~{next_in:.0f}s{RST}")
    print(f"{'─'*58}")
    rc = G if total_ret >= 0 else R
    print(f"  Capital    : {W}${capital:>10,.2f}{RST}  ({rc}{total_ret:+.2f}%{RST})")
    print(f"  Trades     : {n}   WR: {wr:.1f}%   PF: {pf:.2f}")
    print(f"{'─'*58}")

    if active:
        sig_str = f"{G}LONG{RST}"  if active['sig'] == BUY else f"{R}SHORT{RST}"
        rpu     = abs(active['entry_px'] - active['sl'])
        move    = (btc_price - active['entry_px']) if active['sig'] == BUY \
                  else (active['entry_px'] - btc_price)
        upnl    = active['risk_usd'] * move / rpu if rpu > 0 else 0
        uc      = G if upnl >= 0 else R
        elapsed = (now_utc() - active['entry_ts']).total_seconds() / 60
        print(f"  {W}OPEN TRADE{RST}  {sig_str}  ({elapsed:.0f} min ago)")
        print(f"    Entry  : ${active['entry_px']:,.2f}   Risk: ${active['risk_usd']:.2f}")
        print(f"    SL     : {R}${active['sl']:,.2f}{RST}   "
              f"Target : {G}${active['tgt']:,.2f}{RST}")
        print(f"    uPnL   : {uc}${upnl:+.2f}{RST}")
    else:
        print(f"  {DIM}No open trade — watching for signal...{RST}")

    print(f"{'─'*58}")
    if trades:
        print(f"  {DIM}Recent trades:{RST}")
        for t in reversed(trades[-5:]):
            col = G if t['pnl'] > 0 else R
            sig = 'BUY ' if t['sig'] == BUY else 'SELL'
            print(f"    {col}{'✓' if t['pnl']>0 else '✗'}{RST} "
                  f"{t['entry_ts'].strftime('%m/%d %H:%M')}  {sig}  "
                  f"{col}${t['pnl']:+.2f}{RST}")
    print(f"{'═'*58}")
    print(f"  {DIM}Ctrl+C to stop  |  {LOG_FILE}{RST}\n")

# ─────────────────────────────────────────────────────────────── #
#  Main loop                                                       #
# ─────────────────────────────────────────────────────────────── #

def run(initial_capital: float):
    capital = initial_capital
    active  = None
    trades  = []

    print(f"\n{'═'*58}")
    print(f"  Forward Test — Consensus ≥{MIN_VOTES}/5  |  ${initial_capital:,.0f} paper capital")
    print(f"  Logging to: {LOG_FILE}")
    print(f"{'═'*58}")
    print("\n  Fetching initial data...")

    btc = fetch_candles('BTC/USDT')
    eth = fetch_candles('ETH/USDT')
    shared = set(btc['datetime']) & set(eth['datetime'])
    btc    = btc[btc['datetime'].isin(shared)].reset_index(drop=True)
    eth    = eth[eth['datetime'].isin(shared)].reset_index(drop=True)
    print(f"  Loaded {len(btc)} candles  |  "
          f"BTC: ${btc['close'].iloc[-1]:,.2f}")
    print(f"  Waiting for next 5m candle close...\n")

    try:
        while True:
            wait      = seconds_to_next_close()
            btc_price = btc['close'].iloc[-1]
            print_status(capital, initial_capital, active, trades, btc_price, wait)
            time.sleep(wait)

            # ── Refresh candles ────────────────────────────────────
            try:
                btc_new = fetch_candles('BTC/USDT')
                eth_new = fetch_candles('ETH/USDT')
                shared  = set(btc_new['datetime']) & set(eth_new['datetime'])
                btc = btc_new[btc_new['datetime'].isin(shared)].reset_index(drop=True)
                eth = eth_new[eth_new['datetime'].isin(shared)].reset_index(drop=True)
            except Exception as e:
                print(f"\n  {Y}⚠ Fetch error: {e} — retrying next candle{RST}")
                continue

            last   = btc.iloc[-1]
            ts     = last['datetime']

            # ── Manage open trade ──────────────────────────────────
            if active is not None:
                exit_px, pnl = check_exit(active, last)
                if pnl is not None:
                    capital += pnl
                    result   = 'WIN' if pnl > 0 else 'LOSS'
                    record   = dict(
                        entry_ts    = active['entry_ts'],
                        exit_ts     = ts,
                        signal      = 'BUY' if active['sig'] == BUY else 'SELL',
                        entry_price = active['entry_px'],
                        exit_price  = exit_px,
                        sl          = active['sl'],
                        target      = active['tgt'],
                        risk_usd    = active['risk_usd'],
                        pnl         = pnl,
                        status      = result,
                        capital     = capital,
                        votes       = active.get('votes', MIN_VOTES),
                    )
                    log_trade(record)
                    trades.append({**record, 'sig': active['sig']})
                    col = G if pnl > 0 else R
                    print(f"\n  {col}{'✓ WIN' if pnl>0 else '✗ LOSS'}  "
                          f"${pnl:+.2f}   Capital: ${capital:.2f}{RST}")
                    active = None
                    time.sleep(3)
                continue   # don't look for new signal while trade is open

            # ── Daily limits ───────────────────────────────────────
            today     = ts.date()
            today_pnl = sum(t['pnl'] for t in trades
                            if t['entry_ts'].date() == today)
            today_n   = sum(1 for t in trades
                            if t['entry_ts'].date() == today)
            if today_n  >= FIXED['max_trades_day']:           continue
            if today_pnl <= -(capital * FIXED['daily_loss_pct']):  continue
            if today_pnl >= capital * FIXED['daily_profit_pct']:   continue

            # ── Check for new signal ───────────────────────────────
            direction, combo = check_signal(btc, eth)
            if direction is not None:
                df    = btc.reset_index(drop=True)
                atr   = _atr(df, FIXED['atr_period'])
                trade = build_trade(combo, df, atr, len(df)-1, capital, direction)
                if trade is not None:
                    active = {**trade, 'entry_ts': ts}
                    sc = G if direction == 'long' else R
                    print(f"\n  {sc}▶ SIGNAL: {'LONG' if direction=='long' else 'SHORT'}  "
                          f"entry=${trade['entry_px']:,.2f}  "
                          f"sl=${trade['sl']:,.2f}  "
                          f"tgt=${trade['tgt']:,.2f}  "
                          f"risk=${trade['risk_usd']:.2f}  "
                          f"votes={combo.get('votes', MIN_VOTES)}{RST}")
                    time.sleep(3)

    except KeyboardInterrupt:
        print(f"\n\n{'═'*58}")
        print(f"  {Y}Stopped.{RST}")
        total_ret = (capital - initial_capital) / initial_capital * 100
        rc = G if total_ret >= 0 else R
        print(f"  Final capital : ${capital:,.2f}  ({rc}{total_ret:+.2f}%{RST})")
        print(f"  Trades        : {len(trades)}")
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            print(f"  Win rate      : {wins/len(trades)*100:.1f}%")
        print(f"  Log           : {LOG_FILE}")
        print(f"{'═'*58}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=1000.0,
                        help='Starting paper capital in USD (default: 1000)')
    args = parser.parse_args()
    run(args.capital)
