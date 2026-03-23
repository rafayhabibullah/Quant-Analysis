"""
Directional Backtest — BTC/ETH 5m
===================================
Separate optimized parameters for LONG and SHORT directions.
Both run in a single pass with shared capital (realistic simulation).

  LONG  : adx_14_20 + stoch_9_3  | slope_th=0.15  atr_sl=1.5  vol=1.2  ETH✓
          Optimizer: n=62  WR=48.4%  PF=1.92  Return=+49.6%  Score=1.9349  (500d)

  SHORT : sma_20_50 + stoch_9_3  | slope_th=0.25  atr_sl=1.5  vol=1.5  ETH✓
          Optimizer: n=21  WR=71.4%  PF=5.15  Return=+42.3%  Score=5.0864

Source: optimize_directional.py — 365d Binance data (33,024 combos)

Run: python backtest_directional.py [--days 60]
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd

from backtest import fetch, align, INITIAL_CAPITAL, compute_metrics, print_report, plot_dashboard
from optimize_directional import (
    compute_signals, get_entry_candle, FIXED,
    _ema, _atr,
    BUY, SELL, NONE_SIG,
)

# ─────────────────────────────────────────────────────────────── #
#  Directional configs (from 365-day optimizer)                    #
# ─────────────────────────────────────────────────────────────── #

LONG_IND1  = {'id': 'adx_14_20',  'type': 'adx',   'period': 14, 'threshold': 20}
LONG_IND2  = {'id': 'stoch_9_3',  'type': 'stoch',  'k': 9, 'd': 3}
LONG_PARAMS = dict(slope_th=0.15, atr_sl=1.5, vol=1.2, use_eth=True)

SHORT_IND1  = {'id': 'sma_20_50',  'type': 'sma_cross', 'fast': 20, 'slow': 50, 'htf': 100}
SHORT_IND2  = {'id': 'stoch_9_3',  'type': 'stoch',     'k': 9, 'd': 3}
SHORT_PARAMS = dict(slope_th=0.25, atr_sl=1.5, vol=1.5, use_eth=True)

# Warmup: SMA50 needs 50 + SMA100 needs 100 + ADX needs ~28 + buffer
WARMUP = 115

# ─────────────────────────────────────────────────────────────── #
#  Single-pass combined runner                                      #
# ─────────────────────────────────────────────────────────────── #

def run_combined_backtest(btc: pd.DataFrame, eth: pd.DataFrame,
                          capital: float = INITIAL_CAPITAL) -> pd.DataFrame:
    """
    Runs LONG and SHORT signals in a single pass with shared capital.
    Only one trade open at a time. LONG has priority if both signal simultaneously.
    """
    fp  = FIXED
    df  = btc.reset_index(drop=True)
    sec = eth.reset_index(drop=True)

    # ── Pre-compute indicators ────────────────────────────────────
    atr     = _atr(df, fp['atr_period'])
    ema9    = _ema(df['close'], 9)
    vol_sma = df['volume'].rolling(fp['vol_period']).mean() if 'volume' in df.columns else None

    long_ind1_sig  = compute_signals(df, LONG_IND1,  atr)   # ADX signal
    long_ind2_sig  = compute_signals(df, LONG_IND2,  atr)   # Stoch signal (long)
    short_ind1_sig = compute_signals(df, SHORT_IND1, atr)   # SMA cross signal
    short_ind2_sig = compute_signals(df, SHORT_IND2, atr)   # Stoch signal (short)

    # ETH correlation (used for SHORT only)
    s_atr   = _atr(sec, fp['atr_period'])
    eth_cfg = {'type': 'ema_cross', 'fast': 9, 'slow': 15, 'htf': 50,
               'slope_th': 0.05, 'slope_lb': 5}
    eth_sig_arr = compute_signals(sec, eth_cfg, s_atr)

    # ── State ─────────────────────────────────────────────────────
    active    = None   # dict when trade is open
    daily_pnl = 0.0
    trd_today = 0
    cur_date  = None
    cons_loss = 0
    cooldown  = 0
    records   = []

    for idx in range(WARMUP, len(df)):
        row  = df.iloc[idx]
        ts   = row['datetime']
        date = ts.date() if hasattr(ts, 'date') else ts

        if cur_date != date:
            cur_date = date; daily_pnl = 0.0; trd_today = 0

        # ── Manage active trade ───────────────────────────────────
        if active is not None:
            hi = row['high']; lo = row['low']
            closed_pnl = None; exit_px = None

            if active['sig'] == BUY:
                if lo <= active['sl']:
                    exit_px    = active['sl']
                    closed_pnl = -active['risk_usd']
                elif hi >= active['tgt']:
                    exit_px    = active['tgt']
                    closed_pnl = active['risk_usd'] * fp['rr_ratio']
            else:
                if hi >= active['sl']:
                    exit_px    = active['sl']
                    closed_pnl = -active['risk_usd']
                elif lo <= active['tgt']:
                    exit_px    = active['tgt']
                    closed_pnl = active['risk_usd'] * fp['rr_ratio']

            if closed_pnl is not None:
                capital   += closed_pnl
                daily_pnl += closed_pnl
                trd_today += 1
                if closed_pnl < 0:
                    cons_loss += 1
                    if cons_loss >= fp['consec_loss_max']:
                        cooldown  = idx + fp['consec_cooldown']
                        cons_loss = 0
                else:
                    cons_loss = 0
                records.append({
                    'entry_time':   active['entry_ts'],
                    'exit_time':    ts,
                    'signal':       'BUY' if active['sig'] == BUY else 'SELL',
                    'entry_price':  active['entry_px'],
                    'exit_price':   exit_px,
                    'stop_loss':    active['sl'],
                    'target':       active['tgt'],
                    'risk_usd':     active['risk_usd'],
                    'pnl_usd':      closed_pnl,
                    'status':       'WIN' if closed_pnl > 0 else 'LOSS',
                    'candle_type':  'pin_bar',
                    'hold_minutes': (ts - active['entry_ts']).total_seconds() / 60,
                    'sl_pct':       abs(active['entry_px'] - active['sl']) / active['entry_px'] * 100,
                })
                active = None
            continue

        # ── Session / risk filters ────────────────────────────────
        if fp['time_filter'] and hasattr(ts, 'hour'):
            h = ts.hour
            if (0 <= h < 4) or h in fp['blocked_hours']:
                continue
        if idx < cooldown:                               continue
        if trd_today >= fp['max_trades_day']:            continue
        if daily_pnl <= -(capital * fp['daily_loss_pct']): continue
        if daily_pnl >= capital * fp['daily_profit_pct']:  continue

        # ── Combined signals ──────────────────────────────────────
        long_ok  = (int(long_ind1_sig[idx])  == BUY  and
                    int(long_ind2_sig[idx])   == BUY)
        short_ok = (int(short_ind1_sig[idx]) == SELL and
                    int(short_ind2_sig[idx])  == SELL)

        if not long_ok and not short_ok:
            continue

        # ── Try LONG ─────────────────────────────────────────────
        if long_ok:
            lb = fp['slope_lookback']
            if idx >= lb:
                avg_atr = atr.iloc[max(0, idx-lb):idx+1].mean()
                slope = ((ema9.iloc[idx] - ema9.iloc[idx-lb]) / (avg_atr * lb)
                         if avg_atr > 0 else 0)
                if slope < LONG_PARAMS['slope_th']:
                    long_ok = False

        if long_ok and LONG_PARAMS['use_eth']:
            s_idx = min(idx, len(sec) - 1)
            eth_s = int(eth_sig_arr[s_idx])
            if eth_s != NONE_SIG and eth_s != BUY:
                long_ok = False

        if long_ok:
            ctype = get_entry_candle(df, ema9, atr, vol_sma, LONG_PARAMS['vol'], idx, BUY)
            if ctype is None:
                long_ok = False

        if long_ok:
            c = df.iloc[idx]; atr_val = atr.iloc[idx]
            risk_usd = capital * fp['risk_pct']
            sl_px = c['low'] - LONG_PARAMS['atr_sl'] * atr_val
            if sl_px >= c['close']:
                long_ok = False
            else:
                rpu = c['close'] - sl_px
                if rpu < 0.05 * atr_val or rpu > fp['max_rr_sl_atr'] * atr_val:
                    long_ok = False
                else:
                    tgt = c['close'] + fp['rr_ratio'] * rpu
                    active = dict(sig=BUY, entry_ts=ts, entry_px=c['close'],
                                  sl=sl_px, tgt=tgt, risk_usd=risk_usd)
                    continue

        # ── Try SHORT ─────────────────────────────────────────────
        if short_ok:
            lb = fp['slope_lookback']
            if idx >= lb:
                avg_atr = atr.iloc[max(0, idx-lb):idx+1].mean()
                slope = ((ema9.iloc[idx] - ema9.iloc[idx-lb]) / (avg_atr * lb)
                         if avg_atr > 0 else 0)
                if slope > -SHORT_PARAMS['slope_th']:
                    short_ok = False

        if short_ok:
            s_idx = min(idx, len(sec) - 1)
            eth_s = int(eth_sig_arr[s_idx])
            if eth_s != NONE_SIG and eth_s != SELL:
                short_ok = False

        if short_ok:
            ctype = get_entry_candle(df, ema9, atr, vol_sma, SHORT_PARAMS['vol'], idx, SELL)
            if ctype is None:
                short_ok = False

        if short_ok:
            c = df.iloc[idx]; atr_val = atr.iloc[idx]
            risk_usd = capital * fp['risk_pct']
            sl_px = c['high'] + SHORT_PARAMS['atr_sl'] * atr_val
            if sl_px <= c['close']:
                short_ok = False
            else:
                rpu = sl_px - c['close']
                if rpu < 0.05 * atr_val or rpu > fp['max_rr_sl_atr'] * atr_val:
                    short_ok = False
                else:
                    tgt = c['close'] - fp['rr_ratio'] * rpu
                    active = dict(sig=SELL, entry_ts=ts, entry_px=c['close'],
                                  sl=sl_px, tgt=tgt, risk_usd=risk_usd)

    # ── Force-close any open trade at end of data ─────────────────
    if active is not None:
        last = df.iloc[-1]
        lc   = last['close']
        rpu  = abs(active['entry_px'] - active['sl'])
        move = (lc - active['entry_px']) if active['sig'] == BUY else (active['entry_px'] - lc)
        pnl  = active['risk_usd'] * move / rpu if rpu > 0 else 0
        records.append({
            'entry_time':   active['entry_ts'],
            'exit_time':    last['datetime'],
            'signal':       'BUY' if active['sig'] == BUY else 'SELL',
            'entry_price':  active['entry_px'],
            'exit_price':   lc,
            'stop_loss':    active['sl'],
            'target':       active['tgt'],
            'risk_usd':     active['risk_usd'],
            'pnl_usd':      pnl,
            'status':       'FORCE_CLOSED',
            'candle_type':  'pin_bar',
            'hold_minutes': (last['datetime'] - active['entry_ts']).total_seconds() / 60,
            'sl_pct':       abs(active['entry_px'] - active['sl']) / active['entry_px'] * 100,
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


# ─────────────────────────────────────────────────────────────── #
#  Main                                                             #
# ─────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=60,
                        help='Days of historical data (default: 60)')
    args = parser.parse_args()

    print('\n' + '═'*60)
    print('  Directional Backtest — BTC/ETH 5m')
    print('  LONG : adx_14_20 + stoch_9_3  (slope=0.15, atr_sl=1.5, vol=1.2)')
    print('  SHORT: sma_20_50 + stoch_9_3  (slope=0.25, atr_sl=1.5, vol=1.5)')
    print('═'*60)
    print('\nDownloading data...')

    btc_raw = fetch('BTC/USDT', days=args.days, interval='5m')
    eth_raw = fetch('ETH/USDT', days=args.days, interval='5m')
    btc, eth = align(btc_raw, eth_raw)
    print(f'  Aligned: {len(btc):,} candles\n')

    print('Running strategy...')
    trades = run_combined_backtest(btc, eth)
    m = compute_metrics(trades, INITIAL_CAPITAL)
    print_report(m, INITIAL_CAPITAL)

    print('Generating charts...')
    plot_dashboard(m, INITIAL_CAPITAL, save_path='backtest_directional.png')
