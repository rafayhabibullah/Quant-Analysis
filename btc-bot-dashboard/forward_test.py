"""
Live Forward Test — BTC/ETH 5m (Dual Strategy)
===============================================
Runs BOTH strategies on real-time Binance market data:
  1. Consensus ≥4/5 (original — from 500-day optimization)
  2. V2 EMA/Stoch 5-Layer (improved — trailing stop, partial TP, adaptive risk)

Paper trading only — no real orders placed.

  Data  : testnet.binance.vision (real prices, simulated account)
  Entry : on candle CLOSE (no look-ahead)

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

# Original consensus strategy imports
from backtest_multi import precompute, combo_fires, build_trade as consensus_build_trade
from optimize_directional import FIXED, BUY, SELL, _atr
from backtest_directional import WARMUP

# V2 strategy imports
from strategy_v2 import (
    CONFIG as V2_CONFIG, BUY as V2_BUY, SELL as V2_SELL, NONE_SIG,
    atr as v2_atr,
    precompute_indicators,
    check_signal as v2_check_signal,
    check_exit as v2_check_exit,
    build_trade as v2_build_trade,
)

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
#  Strategy 1: Consensus ≥4/5 signal checker                      #
# ─────────────────────────────────────────────────────────────── #

def check_signal_consensus(btc: pd.DataFrame, eth: pd.DataFrame):
    """
    Run Consensus ≥MIN_VOTES on the last candle of the rolling window.
    Returns (direction, combo) or (None, None).
    """
    df  = btc.reset_index(drop=True)
    sec = eth.reset_index(drop=True)
    idx = len(df) - 1

    if idx < WARMUP:
        return None, None

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
#  Strategy 2: V2 EMA/Stoch 5-Layer signal checker                #
# ─────────────────────────────────────────────────────────────── #

def check_signal_v2(btc: pd.DataFrame, eth: pd.DataFrame):
    """
    Run V2 5-layer strategy signal check on the last candle.
    Returns (direction, info_dict) or (None, None).
    """
    df  = btc.reset_index(drop=True)
    sec = eth.reset_index(drop=True)

    sig, ctype = v2_check_signal(df, sec)
    if sig is None:
        return None, None

    direction = 'long' if sig == V2_BUY else 'short'
    info = {'candle_type': ctype, 'strategy': 'v2'}
    return direction, info

# ─────────────────────────────────────────────────────────────── #
#  Combined signal checker                                         #
# ─────────────────────────────────────────────────────────────── #

def check_signal(btc: pd.DataFrame, eth: pd.DataFrame):
    """
    Check BOTH strategies. Returns best signal with strategy tag.
    Priority: if both agree, extra confidence. If only one fires, use it.
    Returns (direction, info_dict) or (None, None).
    """
    dir_con, combo_con = check_signal_consensus(btc, eth)
    dir_v2, info_v2    = check_signal_v2(btc, eth)

    if dir_con and dir_v2 and dir_con == dir_v2:
        # Both strategies agree — highest confidence
        info = {
            'strategy': 'both',
            'votes': combo_con.get('votes', MIN_VOTES) if combo_con else 0,
            'candle_type': info_v2.get('candle_type', ''),
            'confidence': 'high',
        }
        return dir_con, info

    if dir_con and dir_v2 and dir_con != dir_v2:
        # Strategies disagree — skip (conflicting signals)
        return None, None

    if dir_v2:
        info = {
            'strategy': 'v2',
            'candle_type': info_v2.get('candle_type', ''),
            'confidence': 'medium',
        }
        return dir_v2, info

    if dir_con:
        info = {
            'strategy': 'consensus',
            'votes': combo_con.get('votes', MIN_VOTES),
            'confidence': 'medium',
        }
        return dir_con, info

    return None, None

# ─────────────────────────────────────────────────────────────── #
#  Trade exit checker (V2 advanced exits for all trades)          #
# ─────────────────────────────────────────────────────────────── #

def check_exit(active: dict, candle: pd.Series, candles_held: int = 0,
               atr_val: float = 0):
    """
    Advanced exit check with trailing stop, partial TP, time exit.
    Falls back to simple SL/TP if trade doesn't have V2 fields.
    Returns (exit_price, pnl) if trade closed, else (None, None).
    """
    # If trade has V2 fields, use advanced exit management
    if 'initial_sl' in active and 'remaining_pct' in active:
        exit_px, pnl, _ = v2_check_exit(active, candle, candles_held, atr_val)
        return exit_px, pnl

    # Fallback: simple SL/TP (for consensus-only trades)
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

    print(f"\n{'='*66}")
    print(f"  {B}Forward Test — Dual Strategy  |  BTC/ETH 5m{RST}")
    print(f"  {DIM}Consensus >=4/5 + V2 EMA/Stoch 5-Layer{RST}")
    print(f"{'='*66}")
    print(f"  BTC Price  : {W}${btc_price:>10,.2f}{RST}   "
          f"UTC {DIM}{now_utc().strftime('%Y-%m-%d %H:%M:%S')}{RST}")
    print(f"  Next check : {DIM}in ~{next_in:.0f}s{RST}")
    print(f"{'─'*66}")
    rc = G if total_ret >= 0 else R
    print(f"  Capital    : {W}${capital:>10,.2f}{RST}  ({rc}{total_ret:+.2f}%{RST})")
    print(f"  Trades     : {n}   WR: {wr:.1f}%   PF: {pf:.2f}")
    print(f"{'─'*66}")

    if active:
        sig_str = f"{G}LONG{RST}" if active['sig'] == BUY else f"{R}SHORT{RST}"
        initial_sl = active.get('initial_sl', active['sl'])
        rpu     = abs(active['entry_px'] - initial_sl)
        move    = (btc_price - active['entry_px']) if active['sig'] == BUY \
                  else (active['entry_px'] - btc_price)
        upnl    = active['risk_usd'] * move / rpu if rpu > 0 else 0
        uc      = G if upnl >= 0 else R

        strat = active.get('strategy', '?')
        be_str = f" {G}[BE]{RST}" if active.get('breakeven_hit') else ""
        tr_str = f" {Y}[TRAIL]{RST}" if active.get('trailing_active') else ""
        pt_str = f" {B}[50%TP]{RST}" if active.get('remaining_pct', 1.0) < 0.99 else ""
        conf = active.get('confidence', '')
        conf_str = f" [{conf.upper()}]" if conf else ""

        print(f"  {W}OPEN TRADE{RST}  {sig_str}{be_str}{tr_str}{pt_str}  "
              f"{DIM}via {strat}{conf_str}{RST}")
        print(f"    Entry  : ${active['entry_px']:,.2f}   Risk: ${active['risk_usd']:.2f}")
        print(f"    SL     : {R}${active['sl']:,.2f}{RST}   "
              f"Target : {G}${active['tgt']:,.2f}{RST}")
        if initial_sl != active.get('sl'):
            print(f"    Orig SL: {DIM}${initial_sl:,.2f}{RST}")
        print(f"    uPnL   : {uc}${upnl:+.2f}{RST}")
    else:
        print(f"  {DIM}No open trade — watching for signal...{RST}")

    print(f"{'─'*66}")
    if trades:
        print(f"  {DIM}Recent trades:{RST}")
        for t in reversed(trades[-5:]):
            col = G if t['pnl'] > 0 else R
            sig = 'BUY ' if t['sig'] == BUY else 'SELL'
            strat = t.get('strategy', '?')
            ts_str = t['entry_ts']
            if hasattr(ts_str, 'strftime'):
                ts_str = ts_str.strftime('%m/%d %H:%M')
            elif isinstance(ts_str, str):
                ts_str = ts_str[:11]
            print(f"    {col}{'V' if t['pnl']>0 else 'X'}{RST} "
                  f"{ts_str}  {sig}  {col}${t['pnl']:+.2f}{RST}  "
                  f"{DIM}[{strat}]{RST}")
    print(f"{'='*66}")
    print(f"  {DIM}Ctrl+C to stop  |  {LOG_FILE}{RST}\n")

# ─────────────────────────────────────────────────────────────── #
#  Main loop                                                       #
# ─────────────────────────────────────────────────────────────── #

def run(initial_capital: float):
    capital = initial_capital
    active  = None
    trades  = []
    trade_history = []      # For V2 adaptive risk
    equity_history = [initial_capital]
    candles_held = 0

    print(f"\n{'='*66}")
    print(f"  Forward Test — Dual Strategy  |  ${initial_capital:,.0f} paper capital")
    print(f"  Strategy 1: Consensus >=4/5")
    print(f"  Strategy 2: V2 EMA/Stoch 5-Layer (trailing, partial TP, adaptive)")
    print(f"  Logging to: {LOG_FILE}")
    print(f"{'='*66}")
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
                print(f"\n  {Y}! Fetch error: {e} — retrying next candle{RST}")
                continue

            last   = btc.iloc[-1]
            ts     = last['datetime']

            # Compute ATR for exit management
            atr_series = v2_atr(btc, V2_CONFIG['atr_period'])
            atr_val = float(atr_series.iloc[-1])

            # ── Manage open trade ──────────────────────────────────
            if active is not None:
                candles_held += 1
                exit_px, pnl = check_exit(active, last, candles_held, atr_val)
                if pnl is not None:
                    capital += pnl
                    result   = 'WIN' if pnl > 0 else 'LOSS'
                    record   = dict(
                        entry_ts     = active['entry_ts'],
                        exit_ts      = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                        signal       = active.get('signal', 'BUY' if active['sig'] == BUY else 'SELL'),
                        entry_price  = active['entry_px'],
                        exit_price   = float(exit_px),
                        sl           = active['sl'],
                        initial_sl   = active.get('initial_sl', active['sl']),
                        target       = active['tgt'],
                        risk_usd     = active['risk_usd'],
                        pnl          = float(pnl),
                        status       = result,
                        capital      = float(capital),
                        strategy     = active.get('strategy', 'consensus'),
                        confidence   = active.get('confidence', ''),
                        candle_type  = active.get('candle_type', ''),
                        breakeven_hit    = active.get('breakeven_hit', False),
                        trailing_active  = active.get('trailing_active', False),
                        remaining_pct    = active.get('remaining_pct', 1.0),
                        votes        = active.get('votes', 0),
                    )
                    log_trade(record)
                    trades.append({**record, 'sig': active['sig']})
                    trade_history.append(record)
                    equity_history.append(capital)
                    col = G if pnl > 0 else R
                    print(f"\n  {col}{'V WIN' if pnl>0 else 'X LOSS'}  "
                          f"${pnl:+.2f}   Capital: ${capital:.2f}  "
                          f"[{active.get('strategy', '?')}]{RST}")
                    active = None
                    candles_held = 0
                    time.sleep(3)
                continue

            # ── Daily limits ───────────────────────────────────────
            today     = ts.date() if hasattr(ts, 'date') else ts
            today_pnl = sum(t['pnl'] for t in trades
                            if hasattr(t.get('entry_ts'), 'date') and
                            t['entry_ts'].date() == today)
            today_n   = sum(1 for t in trades
                            if hasattr(t.get('entry_ts'), 'date') and
                            t['entry_ts'].date() == today)
            if today_n   >= V2_CONFIG['max_trades_day']:     continue
            if today_pnl <= -(capital * V2_CONFIG['daily_loss_pct']):   continue
            if today_pnl >= capital * V2_CONFIG['daily_profit_pct']:    continue

            # ── Check for new signal (dual strategy) ───────────────
            direction, info = check_signal(btc, eth)
            if direction is not None:
                strategy = info.get('strategy', 'consensus')
                df = btc.reset_index(drop=True)

                # Build trade using V2 builder (has advanced fields) regardless of strategy
                trade = v2_build_trade(
                    V2_BUY if direction == 'long' else V2_SELL,
                    df, atr_series, len(df) - 1, capital,
                    trade_history, equity_history,
                )

                if trade is not None:
                    trade['strategy'] = strategy
                    trade['confidence'] = info.get('confidence', '')
                    trade['votes'] = info.get('votes', 0)
                    trade['candle_type'] = info.get('candle_type', '')
                    active = {
                        **trade,
                        'entry_ts': ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                    }
                    candles_held = 0
                    sc = G if direction == 'long' else R
                    print(f"\n  {sc}> SIGNAL: {'LONG' if direction=='long' else 'SHORT'}  "
                          f"entry=${trade['entry_px']:,.2f}  "
                          f"sl=${trade['sl']:,.2f}  "
                          f"tgt=${trade['tgt']:,.2f}  "
                          f"risk=${trade['risk_usd']:.2f}  "
                          f"via={strategy}  "
                          f"conf={info.get('confidence', '?')}{RST}")
                    time.sleep(3)

    except KeyboardInterrupt:
        print(f"\n\n{'='*66}")
        print(f"  {Y}Stopped.{RST}")
        total_ret = (capital - initial_capital) / initial_capital * 100
        rc = G if total_ret >= 0 else R
        print(f"  Final capital : ${capital:,.2f}  ({rc}{total_ret:+.2f}%{RST})")
        print(f"  Trades        : {len(trades)}")
        if trades:
            wins = sum(1 for t in trades if t['pnl'] > 0)
            print(f"  Win rate      : {wins/len(trades)*100:.1f}%")
            # Strategy breakdown
            by_strat = {}
            for t in trades:
                s = t.get('strategy', '?')
                by_strat.setdefault(s, []).append(t)
            for s, st in by_strat.items():
                sw = sum(1 for t in st if t['pnl'] > 0)
                print(f"    {s}: {len(st)} trades, {sw/len(st)*100:.1f}% WR")
        print(f"  Log           : {LOG_FILE}")
        print(f"{'='*66}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capital', type=float, default=1000.0,
                        help='Starting paper capital in USD (default: 1000)')
    args = parser.parse_args()
    run(args.capital)
