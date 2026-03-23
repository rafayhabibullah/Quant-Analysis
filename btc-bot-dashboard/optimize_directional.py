"""
Directional Parameter Optimizer — BTC/ETH 5m
=============================================
Finds the best indicator combos and parameters SEPARATELY for LONG and SHORT trades.

Logic: runs the full indicator grid twice — once filtering to BUY-only trades,
once filtering to SELL-only trades — so each direction gets its own optimal setup.

Outputs:
  optimize_long.csv       — full results for long direction
  optimize_short.csv      — full results for short direction
  optimize_directional.png — comparison charts

Run: python optimize_directional.py
"""

import warnings
warnings.filterwarnings("ignore")

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from multiprocessing import Pool, cpu_count

from backtest import fetch, align, INITIAL_CAPITAL

# ─────────────────────────────────────────────────────────────── #
#  Signal constants                                                 #
# ─────────────────────────────────────────────────────────────── #

BUY, SELL, NONE_SIG = 1, -1, 0

# ─────────────────────────────────────────────────────────────── #
#  Indicator catalogue  (same as optimize_indicators.py)           #
# ─────────────────────────────────────────────────────────────── #

INDICATOR_CONFIGS = [
    {'id': 'ema_9_15',    'type': 'ema_cross', 'fast': 9,  'slow': 15, 'htf': 50,  'slope_th': 0.15, 'slope_lb': 5},
    {'id': 'ema_9_21',    'type': 'ema_cross', 'fast': 9,  'slow': 21, 'htf': 50,  'slope_th': 0.15, 'slope_lb': 5},
    {'id': 'ema_8_21',    'type': 'ema_cross', 'fast': 8,  'slow': 21, 'htf': 50,  'slope_th': 0.15, 'slope_lb': 5},
    {'id': 'ema_13_34',   'type': 'ema_cross', 'fast': 13, 'slow': 34, 'htf': 100, 'slope_th': 0.10, 'slope_lb': 7},
    {'id': 'sma_5_20',    'type': 'sma_cross', 'fast': 5,  'slow': 20, 'htf': 50},
    {'id': 'sma_10_30',   'type': 'sma_cross', 'fast': 10, 'slow': 30, 'htf': 100},
    {'id': 'sma_20_50',   'type': 'sma_cross', 'fast': 20, 'slow': 50, 'htf': 100},
    {'id': 'macd_12_26_9',    'type': 'macd', 'fast': 12, 'slow': 26, 'sig_p': 9, 'zero_filter': False},
    {'id': 'macd_12_26_9_zl', 'type': 'macd', 'fast': 12, 'slow': 26, 'sig_p': 9, 'zero_filter': True},
    {'id': 'macd_8_17_9',     'type': 'macd', 'fast': 8,  'slow': 17, 'sig_p': 9, 'zero_filter': False},
    {'id': 'rsi_45_55',   'type': 'rsi', 'period': 14, 'bull_th': 45, 'bear_th': 55},
    {'id': 'rsi_50',      'type': 'rsi', 'period': 14, 'bull_th': 50, 'bear_th': 50},
    {'id': 'rsi_40_60',   'type': 'rsi', 'period': 14, 'bull_th': 40, 'bear_th': 60},
    {'id': 'bb_20_2',     'type': 'bb',  'period': 20, 'std_dev': 2.0},
    {'id': 'bb_20_15',    'type': 'bb',  'period': 20, 'std_dev': 1.5},
    {'id': 'adx_14_20',   'type': 'adx', 'period': 14, 'threshold': 20},
    {'id': 'adx_14_25',   'type': 'adx', 'period': 14, 'threshold': 25},
    {'id': 'stoch_14_3',  'type': 'stoch', 'k': 14, 'd': 3},
    {'id': 'stoch_9_3',   'type': 'stoch', 'k': 9,  'd': 3},
]

# Direction-specific risk params (slope_threshold key difference vs generic optimizer)
RISK_PARAM_GRID = {
    'slope_threshold': [0.10, 0.15, 0.20, 0.25],
    'atr_sl_mult':     [0.75, 1.0, 1.5, 2.0],
    'volume_mult':     [1.0, 1.2, 1.5],
}

FIXED = dict(
    risk_pct          = 0.015,
    ema_prox_atr_mult = 1.5,
    max_rr_sl_atr     = 3.0,
    consec_loss_max   = 3,
    consec_cooldown   = 12,
    max_trades_day    = 5,
    daily_loss_pct    = 0.03,
    daily_profit_pct  = 0.04,
    time_filter       = True,
    blocked_hours     = {9, 10, 12, 13, 14, 15},
    rr_ratio          = 2.0,
    atr_period        = 14,
    vol_period        = 20,
    slope_lookback    = 5,
)

# ─────────────────────────────────────────────────────────────── #
#  Math helpers  (identical to optimize_indicators.py)             #
# ─────────────────────────────────────────────────────────────── #

def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def _sma(series, period):
    return series.rolling(period).mean()

def _atr(df, period=14):
    pc = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - pc).abs(),
                    (df['low']  - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def _rsi(close, period=14):
    d    = close.diff()
    gain = d.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _adx_full(df, period=14):
    h, l, c = df['high'], df['low'], df['close']
    pc  = c.shift(1)
    tr  = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    up  = h.diff(); dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr_s = tr.ewm(span=period, adjust=False).mean()
    pdm_s = pd.Series(pdm, index=df.index).ewm(span=period, adjust=False).mean()
    mdm_s = pd.Series(mdm, index=df.index).ewm(span=period, adjust=False).mean()
    pdi   = 100 * pdm_s / atr_s.replace(0, np.nan)
    mdi   = 100 * mdm_s / atr_s.replace(0, np.nan)
    dx    = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(span=period, adjust=False).mean(), pdi, mdi

def _stoch(df, k=14, d=3):
    lo = df['low'].rolling(k).min()
    hi = df['high'].rolling(k).max()
    pct_k = 100 * (df['close'] - lo) / (hi - lo).replace(0, np.nan)
    return pct_k, pct_k.rolling(d).mean()

def compute_signals(df, cfg, atr) -> np.ndarray:
    t = cfg['type']; close = df['close']

    if t == 'ema_cross':
        fe = _ema(close, cfg['fast']); se = _ema(close, cfg['slow'])
        he = _ema(close, cfg['htf']); lb = cfg['slope_lb']
        denom = (_atr(df).rolling(lb).mean() * lb).replace(0, np.nan)
        sf = (fe - fe.shift(lb)) / denom
        ss = (se - se.shift(lb)) / denom
        bull = (fe > se) & (sf > cfg['slope_th']) & (ss > 0) & (close > he)
        bear = (fe < se) & (sf < -cfg['slope_th']) & (ss < 0) & (close < he)

    elif t == 'sma_cross':
        fs = _sma(close, cfg['fast']); ss = _sma(close, cfg['slow'])
        hs = _sma(close, cfg['htf'])
        bull = (fs > ss) & (close > hs) & (fs > fs.shift(1))
        bear = (fs < ss) & (close < hs) & (fs < fs.shift(1))

    elif t == 'macd':
        ml = _ema(close, cfg['fast']) - _ema(close, cfg['slow'])
        sl = _ema(ml, cfg['sig_p']); hist = ml - sl
        bull = (hist > 0) & (ml > 0) if cfg.get('zero_filter') else hist > 0
        bear = (hist < 0) & (ml < 0) if cfg.get('zero_filter') else hist < 0

    elif t == 'rsi':
        rsi = _rsi(close, cfg['period'])
        bull = rsi > cfg['bull_th']; bear = rsi < cfg['bear_th']

    elif t == 'bb':
        mid = _sma(close, cfg['period'])
        bull = close > mid; bear = close < mid

    elif t == 'adx':
        adx_v, pdi, mdi = _adx_full(df, cfg['period'])
        trend = adx_v > cfg['threshold']
        bull = trend & (pdi > mdi); bear = trend & (mdi > pdi)

    elif t == 'stoch':
        pct_k, pct_d = _stoch(df, cfg['k'], cfg['d'])
        bull = (pct_k > 50) & (pct_k > pct_d)
        bear = (pct_k < 50) & (pct_k < pct_d)

    else:
        return np.zeros(len(df), dtype=int)

    sig = np.where(bull.fillna(False), BUY,
                   np.where(bear.fillna(False), SELL, NONE_SIG))
    return sig.astype(int)

# ─────────────────────────────────────────────────────────────── #
#  Candle pattern helpers  (pin bar only — engulfing/reclaim off)  #
# ─────────────────────────────────────────────────────────────── #

def _near_ema(c, ema9_i, atr_i, sig):
    band = atr_i * FIXED['ema_prox_atr_mult']
    return c['low'] <= ema9_i + band if sig == BUY else c['high'] >= ema9_i - band

def _vol_ok(vol_i, vol_sma_i, volume_mult):
    if vol_sma_i is None or pd.isna(vol_sma_i) or vol_sma_i == 0:
        return True
    return vol_i > volume_mult * vol_sma_i

def _pin_bar(c, p, atr_i, sig):
    rng = c['high'] - c['low']
    if rng < 0.5 * atr_i: return False
    body = abs(c['close'] - c['open'])
    if body > 0.30 * rng: return False
    if sig == BUY:
        lw = min(c['open'], c['close']) - c['low']
        return lw >= 2.5 * body and c['close'] > c['open']
    uw = c['high'] - max(c['open'], c['close'])
    return uw >= 2.5 * body and c['close'] < c['open']

def get_entry_candle(df, ema9, atr, vol_sma, volume_mult, idx, sig):
    c = df.iloc[idx]; atr_i = atr.iloc[idx]
    if not _near_ema(c, ema9.iloc[idx], atr_i, sig): return None
    vs = vol_sma.iloc[idx] if vol_sma is not None else None
    if not _vol_ok(c.get('volume', 1), vs, volume_mult): return None
    if abs(c['close'] - c['open']) < 0.08 * atr_i: return None
    p = df.iloc[idx - 1] if idx > 0 else c
    return 'pin_bar' if _pin_bar(c, p, atr_i, sig) else None

# ─────────────────────────────────────────────────────────────── #
#  Warmup helper                                                    #
# ─────────────────────────────────────────────────────────────── #

def _warmup_for(cfg):
    t = cfg['type']
    if t == 'ema_cross': return max(cfg['htf'], cfg['slow']) + cfg['slope_lb'] + 5
    if t == 'sma_cross': return cfg['htf'] + 5
    if t == 'macd':      return cfg['slow'] + cfg['sig_p'] + 5
    if t in ('rsi','bb'): return cfg['period'] + 5
    if t == 'adx':       return cfg['period'] * 2 + 5
    if t == 'stoch':     return cfg['k'] + cfg['d'] + 5
    return 50

# ─────────────────────────────────────────────────────────────── #
#  Directional combo runner                                         #
# ─────────────────────────────────────────────────────────────── #

def run_combo(btc, eth, config, direction):
    """
    direction: 'long' (BUY only) or 'short' (SELL only)
    All other logic identical to optimize_indicators.py.
    """
    fp  = FIXED
    df  = btc.reset_index(drop=True)
    sec = eth.reset_index(drop=True)
    allowed_sig = BUY if direction == 'long' else SELL

    atr     = _atr(df, fp['atr_period'])
    ema9    = _ema(df['close'], 9)
    vol_sma = df['volume'].rolling(fp['vol_period']).mean() if 'volume' in df.columns else None

    ind1_sig = compute_signals(df, config['ind1'], atr)
    ind2_sig = (compute_signals(df, config['ind2'], atr)
                if config['ind2'] is not None else None)

    if config['use_eth_corr']:
        s_atr   = _atr(sec, fp['atr_period'])
        eth_cfg = {'type': 'ema_cross', 'fast': 9, 'slow': 15, 'htf': 50,
                   'slope_th': 0.05, 'slope_lb': 5}
        eth_sig = compute_signals(sec, eth_cfg, s_atr)
    else:
        eth_sig = None

    warmup = max(55, _warmup_for(config['ind1']),
                 _warmup_for(config['ind2']) if config['ind2'] else 0)

    slope_th  = config['slope_threshold']
    sl_mult   = config['atr_sl_mult']
    vol_mult  = config['volume_mult']
    rp        = fp['risk_pct']

    capital   = INITIAL_CAPITAL
    active    = None
    daily_pnl = 0.0
    trd_today = 0
    cur_date  = None
    cons_loss = 0
    cooldown  = 0
    trades    = []
    n         = len(df)

    for idx in range(warmup, n):
        row  = df.iloc[idx]
        ts   = row['datetime']
        date = ts.date() if hasattr(ts, 'date') else ts

        if cur_date != date:
            cur_date = date; daily_pnl = 0.0; trd_today = 0

        # ── Manage active trade ────────────────────────────────
        if active is not None:
            entry, sl, tgt, sig_dir, risk_usd, rpu = active
            hi, lo = row['high'], row['low']
            if sig_dir == BUY:
                if lo <= sl:
                    pnl = -risk_usd
                    capital += pnl; daily_pnl += pnl; trd_today += 1
                    cons_loss += 1
                    if cons_loss >= fp['consec_loss_max']:
                        cooldown = idx + fp['consec_cooldown']; cons_loss = 0
                    trades.append({'pnl': pnl, 'win': False}); active = None
                elif hi >= tgt:
                    pnl = risk_usd * fp['rr_ratio']
                    capital += pnl; daily_pnl += pnl; trd_today += 1
                    cons_loss = 0
                    trades.append({'pnl': pnl, 'win': True}); active = None
            else:
                if hi >= sl:
                    pnl = -risk_usd
                    capital += pnl; daily_pnl += pnl; trd_today += 1
                    cons_loss += 1
                    if cons_loss >= fp['consec_loss_max']:
                        cooldown = idx + fp['consec_cooldown']; cons_loss = 0
                    trades.append({'pnl': pnl, 'win': False}); active = None
                elif lo <= tgt:
                    pnl = risk_usd * fp['rr_ratio']
                    capital += pnl; daily_pnl += pnl; trd_today += 1
                    cons_loss = 0
                    trades.append({'pnl': pnl, 'win': True}); active = None
            continue

        # ── Filters ───────────────────────────────────────────
        if fp['time_filter'] and hasattr(ts, 'hour'):
            h = ts.hour
            if (0 <= h < 4) or h in fp['blocked_hours']:
                continue
        if idx < cooldown: continue
        if trd_today >= fp['max_trades_day']: continue
        if daily_pnl <= -(capital * fp['daily_loss_pct']): continue
        if daily_pnl >= capital * fp['daily_profit_pct']: continue

        # ── Combined trend signal ──────────────────────────────
        sig = int(ind1_sig[idx])
        if ind2_sig is not None:
            s2 = int(ind2_sig[idx])
            if s2 != sig or sig == NONE_SIG: sig = NONE_SIG
        if sig == NONE_SIG: continue

        # ── Direction gate — only trade allowed direction ──────
        if sig != allowed_sig: continue

        # ── Slope threshold gate (applied on top of indicator) ─
        # For EMA-cross indicators slope_th is baked in; for others
        # we apply an additional EMA9 slope check
        if config['ind1']['type'] != 'ema_cross':
            lb   = fp['slope_lookback']
            if idx >= lb:
                ema9_now  = ema9.iloc[idx]
                ema9_prev = ema9.iloc[idx - lb]
                avg_atr   = atr.iloc[max(0, idx-lb):idx+1].mean()
                slope = (ema9_now - ema9_prev) / (avg_atr * lb) if avg_atr > 0 else 0
                if sig == BUY  and slope < slope_th:  continue
                if sig == SELL and slope > -slope_th: continue

        # ── Candle pattern (pin bar only) ──────────────────────
        ctype = get_entry_candle(df, ema9, atr, vol_sma, vol_mult, idx, sig)
        if ctype is None: continue

        # ── ETH correlation ────────────────────────────────────
        if eth_sig is not None:
            s_idx = min(idx, len(sec) - 1)
            eth_s = int(eth_sig[s_idx])
            if eth_s != NONE_SIG and eth_s != sig: continue

        # ── Build trade ────────────────────────────────────────
        c = df.iloc[idx]; atr_val = atr.iloc[idx]
        risk_usd = capital * rp

        if sig == BUY:
            sl_price = c['low'] - sl_mult * atr_val
            if sl_price >= c['close']: continue
            rpu = c['close'] - sl_price
        else:
            sl_price = c['high'] + sl_mult * atr_val
            if sl_price <= c['close']: continue
            rpu = sl_price - c['close']

        if rpu < 0.05 * atr_val or rpu > fp['max_rr_sl_atr'] * atr_val: continue
        tgt = (c['close'] + fp['rr_ratio'] * rpu if sig == BUY
               else c['close'] - fp['rr_ratio'] * rpu)
        active = (c['close'], sl_price, tgt, sig, risk_usd, rpu)

    # Force-close
    if active is not None:
        entry, sl_price, tgt, sig_dir, risk_usd, rpu = active
        last_close = df.iloc[-1]['close']
        move = (last_close - entry) if sig_dir == BUY else (entry - last_close)
        pnl  = risk_usd * move / rpu if rpu > 0 else 0
        capital += pnl
        trades.append({'pnl': pnl, 'win': pnl > 0})

    return trades, capital

# ─────────────────────────────────────────────────────────────── #
#  Metrics & score                                                  #
# ─────────────────────────────────────────────────────────────── #

def compute_metrics(trades, initial_capital):
    if not trades: return None
    pnls   = [t['pnl'] for t in trades]
    wins   = [t['pnl'] for t in trades if t['win']]
    losses = [t['pnl'] for t in trades if not t['win']]
    n      = len(pnls)
    wr     = 100 * len(wins) / n
    gp     = sum(wins) if wins else 0
    gl     = abs(sum(losses)) if losses else 0
    pf     = gp / gl if gl > 0 else (99.0 if gp > 0 else 0)
    rp     = 100 * sum(pnls) / initial_capital
    ps     = pd.Series(pnls)
    sharpe = ps.mean() / ps.std() * np.sqrt(252) if len(pnls) > 1 and ps.std() > 0 else 0
    cum    = ps.cumsum()
    max_dd = 100 * (cum - cum.cummax()).min() / initial_capital
    return dict(n=n, win_rate=wr, profit_factor=pf, sharpe=sharpe,
                max_dd_pct=max_dd, return_pct=rp,
                expectancy=sum(pnls) / n)

def score(m):
    if m is None:                    return float('-inf')
    if m['n'] < 8:                   return float('-inf')
    if m['n'] > 120:                 return float('-inf')
    if m['win_rate']      < 38.0:    return float('-inf')
    if m['profit_factor'] < 1.05:    return float('-inf')
    if m['max_dd_pct']    < -25.0:   return float('-inf')
    return round(
        m['profit_factor']          * 0.30 +
        (m['win_rate'] / 100)       * 0.25 +
        max(0, m['sharpe'])         * 0.25 +
        (1 / (abs(m['max_dd_pct']) + 0.5)) * 0.20, 6)

# ─────────────────────────────────────────────────────────────── #
#  Config builder                                                   #
# ─────────────────────────────────────────────────────────────── #

def build_all_configs(direction):
    risk_combos = list(itertools.product(
        RISK_PARAM_GRID['slope_threshold'],
        RISK_PARAM_GRID['atr_sl_mult'],
        RISK_PARAM_GRID['volume_mult'],
    ))
    by_type = {}
    for ind in INDICATOR_CONFIGS:
        by_type.setdefault(ind['type'], []).append(ind)
    types = list(by_type.keys())

    configs = []
    for use_eth in [True, False]:
        for ind in INDICATOR_CONFIGS:
            for sl_th, sl_m, vm in risk_combos:
                configs.append(dict(ind1=ind, ind2=None, use_eth_corr=use_eth,
                                    slope_threshold=sl_th, atr_sl_mult=sl_m,
                                    volume_mult=vm, direction=direction))
        for i, t1 in enumerate(types):
            for t2 in types[i + 1:]:
                for ind1 in by_type[t1]:
                    for ind2 in by_type[t2]:
                        for sl_th, sl_m, vm in risk_combos:
                            configs.append(dict(ind1=ind1, ind2=ind2,
                                                use_eth_corr=use_eth,
                                                slope_threshold=sl_th,
                                                atr_sl_mult=sl_m,
                                                volume_mult=vm,
                                                direction=direction))
    return configs

# ─────────────────────────────────────────────────────────────── #
#  Multiprocessing worker                                           #
# ─────────────────────────────────────────────────────────────── #

_btc_global = None
_eth_global = None

def _init_worker(btc_df, eth_df):
    global _btc_global, _eth_global
    _btc_global = btc_df
    _eth_global = eth_df

def _run_single(config):
    try:
        trades, _ = run_combo(_btc_global, _eth_global, config, config['direction'])
        m = compute_metrics(trades, INITIAL_CAPITAL)
        s = score(m)
        result = {
            'direction':       config['direction'],
            'ind1_id':         config['ind1']['id'],
            'ind2_id':         config['ind2']['id'] if config['ind2'] else 'none',
            'ind1_type':       config['ind1']['type'],
            'ind2_type':       config['ind2']['type'] if config['ind2'] else 'none',
            'use_eth_corr':    config['use_eth_corr'],
            'slope_threshold': config['slope_threshold'],
            'atr_sl_mult':     config['atr_sl_mult'],
            'volume_mult':     config['volume_mult'],
            'score':           s,
        }
        if m:
            result.update({k: round(v, 3) for k, v in m.items()})
        else:
            result.update(dict(n=0, win_rate=0, profit_factor=0, sharpe=0,
                               max_dd_pct=0, return_pct=0, expectancy=0))
        return result
    except Exception as e:
        return {'direction': config['direction'],
                'ind1_id': config['ind1']['id'],
                'ind2_id': config['ind2']['id'] if config['ind2'] else 'none',
                'score': float('-inf'), 'error': str(e)}

# ─────────────────────────────────────────────────────────────── #
#  Grid search                                                      #
# ─────────────────────────────────────────────────────────────── #

def run_direction(btc, eth, direction):
    configs = build_all_configs(direction)
    total   = len(configs)
    n_jobs  = max(1, cpu_count() - 1)

    print(f"\n  [{direction.upper()}]  {total:,} combos  |  {n_jobs} workers")

    results = []
    with Pool(n_jobs, initializer=_init_worker, initargs=(btc, eth)) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_single, configs, chunksize=20)):
            results.append(res)
            if (i + 1) % 1000 == 0:
                valid = sum(1 for r in results if r.get('score', float('-inf')) > float('-inf'))
                print(f"  {i+1:>6}/{total}  ({(i+1)/total*100:.0f}%)  valid: {valid}", flush=True)

    df = pd.DataFrame(results).sort_values('score', ascending=False).reset_index(drop=True)
    return df

# ─────────────────────────────────────────────────────────────── #
#  Display                                                          #
# ─────────────────────────────────────────────────────────────── #

def print_top(df, direction, n=5):
    """Show top N unique indicator combos (deduped by ind1+ind2 pair, best params shown)."""
    valid = df[df['score'] > float('-inf')].copy()
    # Deduplicate: keep best-scoring row per unique indicator pair
    valid['_combo'] = valid['ind1_id'] + '|' + valid['ind2_id']
    valid = (valid
             .sort_values('score', ascending=False)
             .drop_duplicates(subset='_combo')
             .head(n)
             .reset_index(drop=True))
    label = 'LONG  (BUY)' if direction == 'long' else 'SHORT (SELL)'
    print(f"\n{'═'*80}")
    print(f"  TOP {n} UNIQUE COMBOS — {label}")
    print(f"{'═'*80}")
    for rank, (_, row) in enumerate(valid.iterrows(), 1):
        ind2 = f" + {row['ind2_id']}" if row['ind2_id'] != 'none' else ''
        eth  = 'ETH✓' if row['use_eth_corr'] else 'noETH'
        print(f"\n  #{rank:<2} {row['ind1_id']}{ind2}  [{eth}]")
        print(f"      Score={row['score']:.4f}  n={int(row['n'])}  "
              f"WR={row['win_rate']:.1f}%  PF={row['profit_factor']:.2f}  "
              f"Sharpe={row['sharpe']:.2f}  MaxDD={row['max_dd_pct']:.1f}%  "
              f"Return={row['return_pct']:+.1f}%")
        print(f"      slope_th={row['slope_threshold']}  "
              f"atr_sl={row['atr_sl_mult']}  vol={row['volume_mult']}")
    print(f"\n{'═'*80}")

def plot_comparison(long_df, short_df, save_path='optimize_directional.png'):
    DARK='#0d1117'; TEXT='#c9d1d9'; GRID='#21262d'
    BLUE='#58a6ff'; GREEN='#3fb950'; RED='#f85149'; GOLD='#e3b341'

    fig = plt.figure(figsize=(18, 12), facecolor=DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    def style(ax, title=''):
        ax.set_facecolor(DARK); ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_color(GRID); ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT); ax.grid(color=GRID, linewidth=0.4, alpha=0.6)
        if title: ax.set_title(title, color=TEXT, fontsize=9, pad=5)

    lv = long_df[long_df['score'] > float('-inf')]
    sv = short_df[short_df['score'] > float('-inf')]

    # 1: Score distributions
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.hist(lv['score'], bins=30, alpha=0.65, color=GREEN, label=f'Long ({len(lv)})', edgecolor=DARK)
    ax0.hist(sv['score'], bins=30, alpha=0.65, color=RED,   label=f'Short ({len(sv)})', edgecolor=DARK)
    ax0.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)
    ax0.set_xlabel('Score'); style(ax0, 'Score Distribution: Long vs Short')

    # 2: Best slope_threshold per direction
    ax1 = fig.add_subplot(gs[0, 1])
    lsl = lv.groupby('slope_threshold')['score'].mean()
    ssl = sv.groupby('slope_threshold')['score'].mean()
    x = np.arange(len(lsl)); w = 0.35
    ax1.bar(x - w/2, lsl.values, w, color=GREEN, alpha=0.8, label='Long')
    ax1.bar(x + w/2, ssl.reindex(lsl.index, fill_value=0).values, w, color=RED, alpha=0.8, label='Short')
    ax1.set_xticks(x); ax1.set_xticklabels([str(v) for v in lsl.index], fontsize=8, color=TEXT)
    ax1.set_xlabel('Slope Threshold'); ax1.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)
    style(ax1, 'Slope Threshold Impact')

    # 3: Best atr_sl_mult per direction
    ax2 = fig.add_subplot(gs[0, 2])
    lam = lv.groupby('atr_sl_mult')['score'].mean()
    sam = sv.groupby('atr_sl_mult')['score'].mean()
    x = np.arange(len(lam)); w = 0.35
    ax2.bar(x - w/2, lam.values, w, color=GREEN, alpha=0.8, label='Long')
    ax2.bar(x + w/2, sam.reindex(lam.index, fill_value=0).values, w, color=RED, alpha=0.8, label='Short')
    ax2.set_xticks(x); ax2.set_xticklabels([str(v) for v in lam.index], fontsize=8, color=TEXT)
    ax2.set_xlabel('ATR SL Multiplier'); ax2.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)
    style(ax2, 'ATR SL Multiplier Impact')

    # 4: Best indicator types — long
    ax3 = fig.add_subplot(gs[1, 0])
    lt = lv.groupby('ind1_type')['score'].mean().sort_values()
    ax3.barh(lt.index, lt.values, color=GREEN, alpha=0.8)
    ax3.set_xlabel('Avg Score'); style(ax3, 'Best Indicator Types — Long')

    # 5: Best indicator types — short
    ax4 = fig.add_subplot(gs[1, 1])
    st = sv.groupby('ind1_type')['score'].mean().sort_values()
    ax4.barh(st.index, st.values, color=RED, alpha=0.8)
    ax4.set_xlabel('Avg Score'); style(ax4, 'Best Indicator Types — Short')

    # 6: Win rate vs return scatter (top 100 each)
    ax5 = fig.add_subplot(gs[1, 2])
    lt100 = lv.head(100); st100 = sv.head(100)
    ax5.scatter(lt100['win_rate'], lt100['return_pct'], c=GREEN, alpha=0.5, s=20, label='Long')
    ax5.scatter(st100['win_rate'], st100['return_pct'], c=RED,   alpha=0.5, s=20, label='Short')
    ax5.axvline(50, color=TEXT, lw=0.6, ls='--', alpha=0.4)
    ax5.set_xlabel('Win Rate (%)'); ax5.set_ylabel('Return (%)')
    ax5.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)
    style(ax5, 'Top 100: Win Rate vs Return')

    fig.suptitle('Directional Optimizer — Long vs Short Parameter Comparison',
                 color=TEXT, fontsize=13, fontweight='bold', y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  Chart saved → {save_path}")

# ─────────────────────────────────────────────────────────────── #
#  Main                                                             #
# ─────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=180,
                        help='Days of historical data to use (default: 180)')
    args = parser.parse_args()
    DAYS = args.days

    print('\n' + '═'*60)
    print('  Directional Optimizer — LONG vs SHORT')
    print('  Blocked hours: 09-10, 12-15 UTC (EU morning)')
    print('  Entry pattern: pin bar only')
    print(f'  Data period  : {DAYS} days')
    print('═'*60)

    print(f'\nDownloading {DAYS}d Binance data...')
    btc_raw = fetch('BTC/USDT', days=DAYS, interval='5m')
    eth_raw = fetch('ETH/USDT', days=DAYS, interval='5m')
    btc, eth = align(btc_raw, eth_raw)
    print(f'  Aligned: {len(btc):,} candles\n')

    # ── LONG ──────────────────────────────────────────────────────
    long_df = run_direction(btc, eth, 'long')
    long_df.to_csv('optimize_long.csv', index=False)
    print(f'  Saved → optimize_long.csv')
    print_top(long_df, 'long', n=10)

    # ── SHORT ─────────────────────────────────────────────────────
    short_df = run_direction(btc, eth, 'short')
    short_df.to_csv('optimize_short.csv', index=False)
    print(f'  Saved → optimize_short.csv')
    print_top(short_df, 'short', n=10)

    # ── Summary comparison ────────────────────────────────────────
    print('\n' + '═'*60)
    print('  PARAMETER COMPARISON SUMMARY')
    print('═'*60)

    for direction, df in [('LONG', long_df), ('SHORT', short_df)]:
        best = df[df['score'] > float('-inf')].iloc[0] if not df[df['score'] > float('-inf')].empty else None
        if best is not None:
            ind2 = f" + {best['ind2_id']}" if best['ind2_id'] != 'none' else ''
            print(f"\n  {direction}:")
            print(f"    Indicators    : {best['ind1_id']}{ind2}")
            print(f"    slope_threshold: {best['slope_threshold']}")
            print(f"    atr_sl_mult   : {best['atr_sl_mult']}")
            print(f"    volume_mult   : {best['volume_mult']}")
            print(f"    use_eth_corr  : {best['use_eth_corr']}")
            print(f"    Score={best['score']:.4f}  Return={best['return_pct']:+.1f}%  "
                  f"WR={best['win_rate']:.1f}%  PF={best['profit_factor']:.2f}")

    plot_comparison(long_df, short_df)
