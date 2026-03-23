"""
Multi-Combo Backtest Comparison — BTC/ETH 5m
=============================================
Compares three combination strategies using the top 5 unique combos
found by optimize_directional.py on 365 days:

  Option A  Consensus   : enter only when ≥ K of 5 combos agree (K=2,3,4)
  Option B  Ranked FB   : try combo #1 → fall back to #2…#5 if needed
  Option C  Portfolio   : 5 paired sub-strategies, $200 each ($1,000 total)

Baselines shown for reference:
  Unified     — backtest.py        (EMA 9/21 + Stoch 14/3)
  Directional — backtest_directional.py  (split long/short params)

Run: python backtest_multi.py [--days 365]
"""

import warnings; warnings.filterwarnings("ignore")
import argparse
import numpy as np
import pandas as pd

from backtest           import fetch, align, INITIAL_CAPITAL, compute_metrics
from backtest_directional import run_combined_backtest
from strategy           import EMAScalpingStrategy
from optimize_directional import (
    compute_signals, get_entry_candle, FIXED,
    _ema, _atr, BUY, SELL, NONE_SIG,
)

# ─────────────────────────────────────────────────────────────── #
#  Top-5 combos per direction  (from 365d optimizer)              #
# ─────────────────────────────────────────────────────────────── #

LONG_COMBOS = [
    dict(rank=1, label='adx_14_20 + stoch_9_3',
         ind1=dict(type='adx',      period=14, threshold=20),
         ind2=dict(type='stoch',    k=9, d=3),
         slope_th=0.15, atr_sl=1.5, vol=1.2, use_eth=False),
    dict(rank=2, label='macd_12_26_9zl + stoch_9_3',
         ind1=dict(type='macd',     fast=12, slow=26, sig_p=9, zero_filter=True),
         ind2=dict(type='stoch',    k=9, d=3),
         slope_th=0.15, atr_sl=1.5, vol=1.2, use_eth=True),
    dict(rank=3, label='ema_9_21 + stoch_9_3',
         ind1=dict(type='ema_cross', fast=9, slow=21, htf=50, slope_th=0.15, slope_lb=5),
         ind2=dict(type='stoch',    k=9, d=3),
         slope_th=0.15, atr_sl=1.5, vol=1.2, use_eth=False),
    dict(rank=4, label='ema_9_15 + stoch_9_3',
         ind1=dict(type='ema_cross', fast=9, slow=15, htf=50, slope_th=0.15, slope_lb=5),
         ind2=dict(type='stoch',    k=9, d=3),
         slope_th=0.20, atr_sl=1.5, vol=1.2, use_eth=False),
    dict(rank=5, label='adx_14_20 + stoch_14_3',
         ind1=dict(type='adx',      period=14, threshold=20),
         ind2=dict(type='stoch',    k=14, d=3),
         slope_th=0.15, atr_sl=1.5, vol=1.2, use_eth=True),
]

SHORT_COMBOS = [
    dict(rank=1, label='sma_20_50 + stoch_9_3',
         ind1=dict(type='sma_cross', fast=20, slow=50, htf=100),
         ind2=dict(type='stoch',     k=9, d=3),
         slope_th=0.25, atr_sl=1.5, vol=1.5, use_eth=True),
    dict(rank=2, label='macd_8_17_9 + stoch_9_3',
         ind1=dict(type='macd',      fast=8, slow=17, sig_p=9, zero_filter=False),
         ind2=dict(type='stoch',     k=9, d=3),
         slope_th=0.25, atr_sl=2.0, vol=1.5, use_eth=True),
    dict(rank=3, label='sma_5_20 + stoch_9_3',
         ind1=dict(type='sma_cross', fast=5, slow=20, htf=50),
         ind2=dict(type='stoch',     k=9, d=3),
         slope_th=0.25, atr_sl=2.0, vol=1.5, use_eth=True),
    dict(rank=4, label='stoch_9_3 (solo)',
         ind1=dict(type='stoch',     k=9, d=3),
         ind2=None,
         slope_th=0.25, atr_sl=2.0, vol=1.5, use_eth=False),
    dict(rank=5, label='rsi_40_60 + stoch_9_3',
         ind1=dict(type='rsi',       period=14, bull_th=40, bear_th=60),
         ind2=dict(type='stoch',     k=9, d=3),
         slope_th=0.25, atr_sl=2.0, vol=1.5, use_eth=False),
]

_ETH_CFG = dict(type='ema_cross', fast=9, slow=15, htf=50, slope_th=0.05, slope_lb=5)
WARMUP   = 115

# ─────────────────────────────────────────────────────────────── #
#  Pre-computation                                                  #
# ─────────────────────────────────────────────────────────────── #

def precompute(df: pd.DataFrame, sec: pd.DataFrame):
    """Compute all indicator signals once for all 10 combos."""
    atr     = _atr(df,  FIXED['atr_period'])
    s_atr   = _atr(sec, FIXED['atr_period'])
    ema9    = _ema(df['close'], 9)
    vol_sma = df['volume'].rolling(FIXED['vol_period']).mean() if 'volume' in df.columns else None
    eth_sig = compute_signals(sec, _ETH_CFG, s_atr)

    def prep(combos):
        result = []
        for c in combos:
            p = dict(c)
            p['ind1_sigs'] = compute_signals(df, c['ind1'], atr)
            p['ind2_sigs'] = (compute_signals(df, c['ind2'], atr)
                              if c['ind2'] is not None else None)
            p['eth_sigs']  = eth_sig if c['use_eth'] else None
            result.append(p)
        return result

    return prep(LONG_COMBOS), prep(SHORT_COMBOS), ema9, atr, vol_sma

# ─────────────────────────────────────────────────────────────── #
#  Signal helpers                                                   #
# ─────────────────────────────────────────────────────────────── #

def combo_fires(p, df, ema9, atr, vol_sma, idx, direction) -> bool:
    """Returns True if precomputed combo p fires a valid entry at idx."""
    fp       = FIXED
    required = BUY if direction == 'long' else SELL

    # Indicator agreement
    s1 = int(p['ind1_sigs'][idx])
    if p['ind2_sigs'] is not None:
        s2 = int(p['ind2_sigs'][idx])
        if s1 != required or s2 != required:
            return False
    elif s1 != required:
        return False

    # EMA9 slope gate (skipped for ema_cross — baked in)
    if p['ind1']['type'] != 'ema_cross':
        lb = fp['slope_lookback']
        if idx >= lb:
            avg_atr = atr.iloc[max(0, idx - lb):idx + 1].mean()
            slope   = ((ema9.iloc[idx] - ema9.iloc[idx - lb]) / (avg_atr * lb)
                       if avg_atr > 0 else 0)
            if direction == 'long'  and slope <  p['slope_th']: return False
            if direction == 'short' and slope > -p['slope_th']: return False

    # Candle pattern (pin bar)
    if get_entry_candle(df, ema9, atr, vol_sma, p['vol'], idx, required) is None:
        return False

    # ETH correlation
    if p['eth_sigs'] is not None:
        eth_s = int(p['eth_sigs'][min(idx, len(p['eth_sigs']) - 1)])
        if eth_s != NONE_SIG and eth_s != required:
            return False

    return True


def build_trade(combo, df, atr, idx, capital, direction) -> dict | None:
    """Build a trade dict using the combo's atr_sl. Returns None if invalid."""
    fp      = FIXED
    c       = df.iloc[idx]
    atr_val = atr.iloc[idx]
    risk    = capital * fp['risk_pct']

    if direction == 'long':
        sl  = c['low']  - combo['atr_sl'] * atr_val
        if sl >= c['close']: return None
        rpu = c['close'] - sl
        tgt = c['close'] + fp['rr_ratio'] * rpu
    else:
        sl  = c['high'] + combo['atr_sl'] * atr_val
        if sl <= c['close']: return None
        rpu = sl - c['close']
        tgt = c['close'] - fp['rr_ratio'] * rpu

    if rpu < 0.05 * atr_val or rpu > fp['max_rr_sl_atr'] * atr_val:
        return None

    return dict(
        sig      = BUY if direction == 'long' else SELL,
        sig_str  = 'BUY' if direction == 'long' else 'SELL',
        entry_ts = df['datetime'].iloc[idx],
        entry_px = c['close'],
        sl=sl, tgt=tgt, risk_usd=risk, rpu=rpu,
        combo_label=combo.get('label', ''),
    )

# ─────────────────────────────────────────────────────────────── #
#  Shared run loop                                                  #
# ─────────────────────────────────────────────────────────────── #

def _run_loop(df, ema9, atr, vol_sma, get_signal_fn, initial_capital) -> pd.DataFrame:
    """
    Core backtest loop. get_signal_fn(idx) -> (direction_str, combo) | (None, None)
    direction_str is 'long' or 'short'.
    """
    fp        = FIXED
    capital   = initial_capital
    active    = None
    daily_pnl = 0.0; trd_today = 0; cur_date = None
    cons_loss = 0;   cooldown  = 0
    records   = []

    for idx in range(WARMUP, len(df)):
        row  = df.iloc[idx]
        ts   = row['datetime']
        date = ts.date() if hasattr(ts, 'date') else ts

        if cur_date != date:
            cur_date = date; daily_pnl = 0.0; trd_today = 0

        # ── Manage open trade ─────────────────────────────────────
        if active is not None:
            hi = row['high']; lo = row['low']
            closed_pnl = None; exit_px = None

            if active['sig'] == BUY:
                if   lo <= active['sl']:  exit_px = active['sl'];  closed_pnl = -active['risk_usd']
                elif hi >= active['tgt']: exit_px = active['tgt']; closed_pnl =  active['risk_usd'] * fp['rr_ratio']
            else:
                if   hi >= active['sl']:  exit_px = active['sl'];  closed_pnl = -active['risk_usd']
                elif lo <= active['tgt']: exit_px = active['tgt']; closed_pnl =  active['risk_usd'] * fp['rr_ratio']

            if closed_pnl is not None:
                capital   += closed_pnl
                daily_pnl += closed_pnl
                trd_today += 1
                if closed_pnl < 0:
                    cons_loss += 1
                    if cons_loss >= fp['consec_loss_max']:
                        cooldown = idx + fp['consec_cooldown']; cons_loss = 0
                else:
                    cons_loss = 0
                records.append(dict(
                    entry_time  = active['entry_ts'], exit_time  = ts,
                    signal      = active['sig_str'],
                    entry_price = active['entry_px'],  exit_price = exit_px,
                    stop_loss   = active['sl'],         target     = active['tgt'],
                    risk_usd    = active['risk_usd'],   pnl_usd    = closed_pnl,
                    status      = 'WIN' if closed_pnl > 0 else 'LOSS',
                    candle_type = 'pin_bar',
                    hold_minutes= (ts - active['entry_ts']).total_seconds() / 60,
                    sl_pct      = abs(active['entry_px'] - active['sl']) / active['entry_px'] * 100,
                    combo       = active['combo_label'],
                    votes       = active.get('votes', 1),
                ))
                active = None
            continue

        # ── Session / risk filters ────────────────────────────────
        if fp['time_filter'] and hasattr(ts, 'hour'):
            h = ts.hour
            if (0 <= h < 4) or h in fp['blocked_hours']: continue
        if idx < cooldown:                                  continue
        if trd_today >= fp['max_trades_day']:               continue
        if daily_pnl <= -(capital * fp['daily_loss_pct']):  continue
        if daily_pnl >= capital * fp['daily_profit_pct']:   continue

        # ── Get signal (option-specific) ──────────────────────────
        direction, combo = get_signal_fn(idx)
        if direction is None: continue

        # ── Build and open trade ──────────────────────────────────
        t = build_trade(combo, df, atr, idx, capital, direction)
        if t is None: continue
        active = t

    # ── Force-close at end of data ────────────────────────────────
    if active is not None:
        last = df.iloc[-1]; lc = last['close']
        move = (lc - active['entry_px']) if active['sig'] == BUY else (active['entry_px'] - lc)
        pnl  = active['risk_usd'] * move / active['rpu'] if active['rpu'] > 0 else 0
        records.append(dict(
            entry_time  = active['entry_ts'], exit_time  = last['datetime'],
            signal      = active['sig_str'],
            entry_price = active['entry_px'],  exit_price = lc,
            stop_loss   = active['sl'],         target     = active['tgt'],
            risk_usd    = active['risk_usd'],   pnl_usd    = pnl,
            status      = 'FORCE_CLOSED', candle_type = 'pin_bar',
            hold_minutes= (last['datetime'] - active['entry_ts']).total_seconds() / 60,
            sl_pct      = abs(active['entry_px'] - active['sl']) / active['entry_px'] * 100,
            combo       = active['combo_label'],
            votes       = active.get('votes', 1),
        ))

    return pd.DataFrame(records) if records else pd.DataFrame()

# ─────────────────────────────────────────────────────────────── #
#  Option A — Consensus                                             #
# ─────────────────────────────────────────────────────────────── #

def run_option_a(df, sec, long_pre, short_pre, ema9, atr, vol_sma,
                 min_votes=3, capital=INITIAL_CAPITAL) -> pd.DataFrame:
    """Enter only when ≥ min_votes of 5 combos agree. Uses best-ranked voter's params."""

    def get_signal(idx):
        long_voters  = [p for p in long_pre  if combo_fires(p, df, ema9, atr, vol_sma, idx, 'long')]
        short_voters = [p for p in short_pre if combo_fires(p, df, ema9, atr, vol_sma, idx, 'short')]
        if len(long_voters) >= min_votes:
            c = long_voters[0]; c['votes'] = len(long_voters)
            return 'long', c
        if len(short_voters) >= min_votes:
            c = short_voters[0]; c['votes'] = len(short_voters)
            return 'short', c
        return None, None

    trades = _run_loop(df, ema9, atr, vol_sma, get_signal, capital)
    return trades

# ─────────────────────────────────────────────────────────────── #
#  Option B — Ranked Fallback                                       #
# ─────────────────────────────────────────────────────────────── #

def run_option_b(df, sec, long_pre, short_pre, ema9, atr, vol_sma) -> pd.DataFrame:
    """Try combo #1 first; fall back to #2…#5 in quality order."""

    def get_signal(idx):
        for p in long_pre:
            if combo_fires(p, df, ema9, atr, vol_sma, idx, 'long'):
                return 'long', p
        for p in short_pre:
            if combo_fires(p, df, ema9, atr, vol_sma, idx, 'short'):
                return 'short', p
        return None, None

    return _run_loop(df, ema9, atr, vol_sma, get_signal, INITIAL_CAPITAL)

# ─────────────────────────────────────────────────────────────── #
#  Option C — Portfolio                                             #
# ─────────────────────────────────────────────────────────────── #

def run_option_c(df, sec, long_pre, short_pre, ema9, atr, vol_sma) -> pd.DataFrame:
    """
    5 paired sub-strategies, $200 each ($1,000 total).
    Pair N uses long combo #N for longs and short combo #N for shorts.
    Trades run independently — multiple open positions possible simultaneously.
    """
    sub_capital = INITIAL_CAPITAL / len(LONG_COMBOS)
    all_trades  = []

    for i in range(len(LONG_COMBOS)):
        lp = long_pre[i]
        sp = short_pre[i]

        def make_signal(lp=lp, sp=sp):
            def get_signal(idx):
                if combo_fires(lp, df, ema9, atr, vol_sma, idx, 'long'):
                    return 'long', lp
                if combo_fires(sp, df, ema9, atr, vol_sma, idx, 'short'):
                    return 'short', sp
                return None, None
            return get_signal

        trades = _run_loop(df, ema9, atr, vol_sma, make_signal(), sub_capital)
        if not trades.empty:
            trades['sub'] = i + 1
            all_trades.append(trades)

    if not all_trades:
        return pd.DataFrame()

    return (pd.concat(all_trades, ignore_index=True)
              .sort_values('entry_time')
              .reset_index(drop=True))

# ─────────────────────────────────────────────────────────────── #
#  Metrics summary                                                  #
# ─────────────────────────────────────────────────────────────── #

def quick_stats(tdf: pd.DataFrame, initial_capital: float) -> dict:
    """Compute key stats for comparison table."""
    if tdf is None or tdf.empty:
        return dict(ret=0, wr=0, pf=0, dd=0, sharpe=0, n=0, long_wr=0, short_wr=0,
                    long_n=0, short_n=0)

    pnls   = tdf['pnl_usd']
    wins   = tdf[tdf['pnl_usd'] > 0]
    losses = tdf[tdf['pnl_usd'] <= 0]
    n      = len(tdf)
    wr     = len(wins) / n * 100
    gp     = wins['pnl_usd'].sum()
    gl     = abs(losses['pnl_usd'].sum()) if len(losses) else 1e-9
    pf     = gp / gl

    equity = initial_capital + pnls.cumsum()
    peak   = equity.cummax()
    dd     = ((equity - peak) / peak * 100).min()
    ret    = pnls.sum() / initial_capital * 100

    tdf2        = tdf.copy()
    tdf2['date'] = pd.to_datetime(tdf2['entry_time']).dt.date
    daily        = tdf2.groupby('date')['pnl_usd'].sum()
    sharpe       = (daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0

    longs  = tdf[tdf['signal'] == 'BUY']
    shorts = tdf[tdf['signal'] == 'SELL']
    long_wr  = (longs['pnl_usd'] > 0).mean() * 100  if len(longs)  else 0
    short_wr = (shorts['pnl_usd'] > 0).mean() * 100 if len(shorts) else 0

    return dict(ret=ret, wr=wr, pf=pf, dd=dd, sharpe=sharpe, n=n,
                long_wr=long_wr, short_wr=short_wr,
                long_n=len(longs), short_n=len(shorts))

# ─────────────────────────────────────────────────────────────── #
#  Comparison table                                                 #
# ─────────────────────────────────────────────────────────────── #

def print_comparison(results: dict, days: int):
    W = '═' * 104
    D = '─' * 104
    print(f'\n{W}')
    print(f'  MULTI-COMBO BACKTEST COMPARISON  —  {days} days, ${INITIAL_CAPITAL:,.0f} starting capital')
    print(f'{W}')
    print(f"  {'Strategy':<32} {'Return':>8} {'WR':>7} {'PF':>6} {'MaxDD':>7} {'Sharpe':>7} "
          f"{'Trades':>7} {'Long':>7} {'n_L':>5} {'Short':>7} {'n_S':>5}")
    print(D)

    sections = [
        ('── Baselines ──', ['unified', 'directional']),
        ('── Option A: Consensus ──', ['a_votes2', 'a_votes3', 'a_votes4']),
        ('── Option B: Ranked Fallback ──', ['b']),
        ('── Option C: Portfolio (5 pairs × $200) ──', ['c']),
    ]

    labels = {
        'unified':     'Unified  (EMA9/21 + Stoch14/3)',
        'directional': 'Directional  (split params)',
        'a_votes2':    'A — Consensus  (≥2 / 5 votes)',
        'a_votes3':    'A — Consensus  (≥3 / 5 votes)',
        'a_votes4':    'A — Consensus  (≥4 / 5 votes)',
        'b':           'B — Ranked Fallback',
        'c':           'C — Portfolio  (5 sub-strategies)',
    }

    for section_title, keys in sections:
        print(f'\n  {section_title}')
        for k in keys:
            if k not in results: continue
            s = results[k]
            print(f"  {labels[k]:<32} "
                  f"{s['ret']:>+7.1f}%"
                  f"  {s['wr']:>5.1f}%"
                  f"  {s['pf']:>5.2f}"
                  f"  {s['dd']:>6.1f}%"
                  f"  {s['sharpe']:>6.2f}"
                  f"  {s['n']:>6}"
                  f"  {s['long_wr']:>5.1f}%  {s['long_n']:>4}"
                  f"  {s['short_wr']:>5.1f}%  {s['short_n']:>4}")

    print(f'\n{W}')
    print(f"  Columns: Return | Win Rate | Profit Factor | Max Drawdown | Sharpe | "
          f"Total Trades | Long WR (n) | Short WR (n)")
    print(f'{W}\n')

# ─────────────────────────────────────────────────────────────── #
#  Per-combo vote breakdown  (Option A)                            #
# ─────────────────────────────────────────────────────────────── #

def print_vote_breakdown(trades_a2, trades_a3, trades_a4):
    print(f"\n{'═'*60}")
    print(f"  Option A — Vote Distribution")
    print(f"{'═'*60}")
    for label, tdf in [('≥2 votes', trades_a2), ('≥3 votes', trades_a3), ('≥4 votes', trades_a4)]:
        if tdf.empty: continue
        vc = tdf['votes'].value_counts().sort_index()
        total = len(tdf)
        parts = [f"{v}v: {c} ({c/total*100:.0f}%)" for v, c in vc.items()]
        wr_by_votes = tdf.groupby('votes').apply(lambda x: (x['pnl_usd'] > 0).mean() * 100)
        wr_parts    = [f"WR@{v}v={w:.0f}%" for v, w in wr_by_votes.items()]
        print(f"  {label:<12}  total={total}  [{', '.join(parts)}]  [{', '.join(wr_parts)}]")

# ─────────────────────────────────────────────────────────────── #
#  Option C per-sub-strategy breakdown                             #
# ─────────────────────────────────────────────────────────────── #

def print_portfolio_breakdown(trades_c):
    if trades_c.empty or 'sub' not in trades_c.columns: return
    sub_cap = INITIAL_CAPITAL / len(LONG_COMBOS)
    print(f"\n{'═'*80}")
    print(f"  Option C — Per Sub-Strategy Breakdown  (${sub_cap:.0f} each)")
    print(f"{'═'*80}")
    print(f"  {'Sub':>4}  {'Long combo':<26}  {'Short combo':<26}  "
          f"{'n':>4}  {'WR':>6}  {'Ret':>8}")
    print(f"  {'─'*78}")
    for i in range(len(LONG_COMBOS)):
        sub = i + 1
        st  = trades_c[trades_c['sub'] == sub]
        if st.empty: continue
        n   = len(st)
        wr  = (st['pnl_usd'] > 0).mean() * 100
        ret = st['pnl_usd'].sum() / sub_cap * 100
        ll  = LONG_COMBOS[i]['label'][:24]
        sl  = SHORT_COMBOS[i]['label'][:24]
        print(f"  {sub:>4}  {ll:<26}  {sl:<26}  {n:>4}  {wr:>5.1f}%  {ret:>+7.1f}%")

# ─────────────────────────────────────────────────────────────── #
#  Main                                                             #
# ─────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=365,
                        help='Days of historical data (default: 365)')
    args = parser.parse_args()

    print(f'\n{"═"*60}')
    print(f'  Multi-Combo Backtest — {args.days} days')
    print(f'{"═"*60}')
    print('\nDownloading data...')

    btc_raw = fetch('BTC/USDT', days=args.days, interval='5m')
    eth_raw = fetch('ETH/USDT', days=args.days, interval='5m')
    btc, eth = align(btc_raw, eth_raw)
    print(f'  Aligned: {len(btc):,} candles\n')

    df  = btc.reset_index(drop=True)
    sec = eth.reset_index(drop=True)

    # ── Pre-compute all signals once ─────────────────────────────
    print('Pre-computing signals for all 10 combos...')
    long_pre, short_pre, ema9, atr, vol_sma = precompute(df, sec)
    print('  Done.\n')

    results = {}

    # ── Baselines ─────────────────────────────────────────────────
    print('Running baseline: Unified (EMA 9/21 + Stoch 14/3)...')
    strat = EMAScalpingStrategy(
        capital=INITIAL_CAPITAL, risk_pct=0.015,
        ema_fast=9, ema_slow=21, ema_htf=50, atr_period=14,
        atr_sl_mult=1.5, slope_threshold=0.15, slope_lookback=5,
        ema_prox_pct=0.003, volume_mult=1.2, volume_period=20,
        max_rr_sl_atr=3.0, consec_loss_max=3, consec_cooldown=12,
        max_trades_day=5, daily_loss_pct=0.03, daily_profit_pct=0.04,
        time_filter=True, rr_ratio=2.0,
        use_stoch=True, stoch_k=14, stoch_d=3,
        use_ema_reclaim=False, use_engulfing=False,
        blocked_hours={9, 10, 12, 13, 14, 15},
    )
    t_unified = strat.run(btc, eth)
    results['unified'] = quick_stats(t_unified, INITIAL_CAPITAL)

    print('Running baseline: Directional (split params)...')
    t_dir = run_combined_backtest(btc, eth)
    results['directional'] = quick_stats(t_dir, INITIAL_CAPITAL)

    # ── Option A — Consensus (three thresholds) ───────────────────
    for votes in [2, 3, 4]:
        print(f'Running Option A: Consensus (≥{votes}/5 votes)...')
        t = run_option_a(df, sec, long_pre, short_pre, ema9, atr, vol_sma, min_votes=votes)
        results[f'a_votes{votes}'] = quick_stats(t, INITIAL_CAPITAL)

    # Store for breakdown
    trades_a2 = run_option_a(df, sec, long_pre, short_pre, ema9, atr, vol_sma, min_votes=2)
    trades_a3 = run_option_a(df, sec, long_pre, short_pre, ema9, atr, vol_sma, min_votes=3)
    trades_a4 = run_option_a(df, sec, long_pre, short_pre, ema9, atr, vol_sma, min_votes=4)

    # ── Option B — Ranked Fallback ────────────────────────────────
    print('Running Option B: Ranked Fallback...')
    trades_b = run_option_b(df, sec, long_pre, short_pre, ema9, atr, vol_sma)
    results['b'] = quick_stats(trades_b, INITIAL_CAPITAL)

    # ── Option C — Portfolio ──────────────────────────────────────
    print('Running Option C: Portfolio (5 sub-strategies)...')
    trades_c = run_option_c(df, sec, long_pre, short_pre, ema9, atr, vol_sma)
    results['c'] = quick_stats(trades_c, INITIAL_CAPITAL)

    # ── Print results ─────────────────────────────────────────────
    print_comparison(results, args.days)
    print_vote_breakdown(trades_a2, trades_a3, trades_a4)
    print_portfolio_breakdown(trades_c)
