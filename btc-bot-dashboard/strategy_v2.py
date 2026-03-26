"""
Strategy V2 — EMA 9/21 + Stochastic Scalping with Advanced Exits
=================================================================
Ported from trading-system/agents/strategy_agent.py.
Self-contained: no external config imports.

5-Layer Entry Filter:
  1. HTF Regime (1h EMA20/50) — only trade with the trend
  2. Trend Detection (EMA9/21 slope + EMA50 bias)
  3. Pin Bar / Engulfing pattern near EMA zone
  4. Stochastic(14,3) confirmation
  5. Correlation filter (ETH must not oppose BTC)

Advanced Exits:
  - Trailing stop (breakeven at +1R, trail at 1.5 ATR after +1.5R)
  - Partial take-profit (50% at +1R)
  - Time exit (60 candles max hold)

Adaptive Risk:
  - 3%/2.5%/1.5% based on rolling 20-trade win rate
  - Equity curve filter (halves size when equity < EMA)
  - ATR expansion filter
"""

import numpy as np
import pandas as pd
from typing import Optional


# ── Signal Constants ─────────────────────────────────────────────
BUY, SELL, NONE_SIG = 1, -1, 0

# ── Strategy Configuration ───────────────────────────────────────
CONFIG = dict(
    # EMA
    ema_fast=9,
    ema_slow=21,
    ema_htf=50,
    # ATR
    atr_period=14,
    atr_sl_mult=0.75,
    # Slope
    slope_threshold=0.25,
    slope_lookback=5,
    # Volume
    volume_mult=1.3,
    volume_period=20,
    # Risk
    risk_pct=0.025,
    max_rr_sl_atr=3.0,
    rr_ratio=2.0,
    initial_capital=1000,
    max_trades_day=8,
    daily_loss_pct=0.05,
    daily_profit_pct=0.08,
    consec_loss_max=3,
    consec_cooldown=12,
    # Stochastic
    use_stoch=True,
    stoch_k=14,
    stoch_d=3,
    # Patterns
    use_pin_bar=True,
    use_engulfing=False,
    use_ema_reclaim=False,
    # Time filter
    blocked_hours={0, 1, 2, 3},
    # Trailing stop
    trailing_stop=True,
    breakeven_at_r=1.0,
    trail_after_r=1.5,
    trail_atr_mult=1.5,
    # Partial TP
    partial_tp=True,
    partial_tp_pct=0.5,
    partial_tp_at_r=1.0,
    # Time exit
    max_hold_candles=60,
    # Adaptive risk
    adaptive_risk=True,
    adaptive_lookback=20,
    risk_pct_high=0.03,
    risk_pct_med=0.025,
    risk_pct_low=0.015,
    # ATR expansion
    atr_expansion_filter=True,
    atr_expansion_ratio=0.8,
    # Equity curve
    equity_curve_filter=True,
    equity_ema_period=20,
    # HTF regime
    regime_ema_fast=20,
    regime_ema_slow=50,
    allow_counter_trend=False,
)


# ── Indicators ───────────────────────────────────────────────────

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def stoch(df: pd.DataFrame, k: int = 14, d: int = 3):
    lo = df['low'].rolling(k).min()
    hi = df['high'].rolling(k).max()
    pct_k = 100 * (df['close'] - lo) / (hi - lo).replace(0, np.nan)
    pct_d = pct_k.rolling(d).mean()
    return pct_k, pct_d


def slope_normalized(ema_s: pd.Series, atr_s: pd.Series, idx: int,
                     lookback: int = 5) -> float:
    if idx < lookback:
        return 0.0
    price_change = ema_s.iloc[idx] - ema_s.iloc[idx - lookback]
    avg_atr = atr_s.iloc[max(0, idx - lookback): idx + 1].mean()
    if avg_atr == 0:
        return 0.0
    return price_change / (avg_atr * lookback)


# ── Trend Detection ──────────────────────────────────────────────

def detect_trend(df, ema9, ema21, ema50, atr_s, idx, cfg=CONFIG):
    s9 = slope_normalized(ema9, atr_s, idx, cfg['slope_lookback'])
    s21 = slope_normalized(ema21, atr_s, idx, cfg['slope_lookback'])
    price = df['close'].iloc[idx]

    bull = (ema9.iloc[idx] > ema21.iloc[idx]
            and s9 > cfg['slope_threshold']
            and s21 > 0
            and price > ema50.iloc[idx])
    bear = (ema9.iloc[idx] < ema21.iloc[idx]
            and s9 < -cfg['slope_threshold']
            and s21 < 0
            and price < ema50.iloc[idx])

    if bull:
        return BUY
    if bear:
        return SELL
    return NONE_SIG


# ── Entry Candle Patterns ────────────────────────────────────────

def _near_ema(df, ema9, atr_s, idx, sig):
    c = df.iloc[idx]
    band = atr_s.iloc[idx] * 1.5
    if sig == BUY:
        return c['low'] <= ema9.iloc[idx] + band
    return c['high'] >= ema9.iloc[idx] - band


def _vol_ok(df, vol_sma, idx, cfg=CONFIG):
    if vol_sma is None:
        return True
    ref = vol_sma.iloc[idx]
    if pd.isna(ref) or ref == 0:
        return True
    return df['volume'].iloc[idx] > cfg['volume_mult'] * ref


def _is_pin_bar(df, atr_s, idx, sig):
    c = df.iloc[idx]
    rng = c['high'] - c['low']
    if rng < 0.5 * atr_s.iloc[idx]:
        return False
    body = abs(c['close'] - c['open'])
    if body > 0.30 * rng:
        return False
    if sig == BUY:
        lower_wick = min(c['open'], c['close']) - c['low']
        return lower_wick >= 2.5 * body and c['close'] > c['open']
    upper_wick = c['high'] - max(c['open'], c['close'])
    return upper_wick >= 2.5 * body and c['close'] < c['open']


def _is_engulfing(df, atr_s, idx, sig):
    if idx < 1:
        return False
    c = df.iloc[idx]
    p = df.iloc[idx - 1]
    curr_body = abs(c['close'] - c['open'])
    prev_body = abs(p['close'] - p['open'])
    if curr_body < 0.5 * atr_s.iloc[idx]:
        return False
    if prev_body == 0 or curr_body < 1.5 * prev_body:
        return False
    if sig == BUY:
        return (p['close'] < p['open'] and c['close'] > c['open']
                and c['open'] < p['close'] and c['close'] > p['open'])
    return (p['close'] > p['open'] and c['close'] < c['open']
            and c['open'] > p['close'] and c['close'] < p['open'])


def entry_candle(df, ema9, ema21, atr_s, vol_sma, idx, sig, cfg=CONFIG):
    if not _near_ema(df, ema9, atr_s, idx, sig):
        return None
    if not _vol_ok(df, vol_sma, idx, cfg):
        return None
    c = df.iloc[idx]
    if abs(c['close'] - c['open']) < 0.08 * atr_s.iloc[idx]:
        return None
    if cfg['use_pin_bar'] and _is_pin_bar(df, atr_s, idx, sig):
        return 'pin_bar'
    if cfg['use_engulfing'] and _is_engulfing(df, atr_s, idx, sig):
        return 'engulfing'
    if cfg['use_ema_reclaim']:
        if idx >= 1:
            c_ = df.iloc[idx]
            p_ = df.iloc[idx - 1]
            body = abs(c_['close'] - c_['open'])
            if body >= 0.4 * atr_s.iloc[idx]:
                if sig == BUY:
                    if (p_['low'] < ema21.iloc[idx - 1]
                            and c_['close'] > ema9.iloc[idx]
                            and c_['close'] > c_['open']):
                        return 'ema_reclaim'
                else:
                    if (p_['high'] > ema21.iloc[idx - 1]
                            and c_['close'] < ema9.iloc[idx]
                            and c_['close'] < c_['open']):
                        return 'ema_reclaim'
    return None


# ── Stochastic Confirmation ──────────────────────────────────────

def stoch_confirms(pct_k, pct_d, idx, sig):
    k = pct_k.iloc[idx]
    d = pct_d.iloc[idx]
    if pd.isna(k) or pd.isna(d):
        return False
    if sig == BUY:
        return k > 50 and k > d
    return k < 50 and k < d


# ── Correlation Filter ───────────────────────────────────────────

def corr_confirms(sig, sec_df, sec_ema9, sec_ema21, sec_ema50, sec_atr, idx, cfg=CONFIG):
    sec_sig = detect_trend(sec_df, sec_ema9, sec_ema21, sec_ema50, sec_atr, idx, cfg)
    if sec_sig == NONE_SIG:
        return True
    return sec_sig == sig


# ── HTF Regime ───────────────────────────────────────────────────

def compute_htf_regime(df_5m, cfg=CONFIG):
    df_h = df_5m.set_index('datetime').resample('1h').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna().reset_index()

    if len(df_h) < 60:
        return {}

    ema_f = ema(df_h['close'], cfg['regime_ema_fast'])
    ema_s = ema(df_h['close'], cfg['regime_ema_slow'])

    regime_map = {}
    h_idx = 0
    for i in range(len(df_5m)):
        ts = df_5m['datetime'].iloc[i]
        while h_idx < len(df_h) - 1 and df_h['datetime'].iloc[h_idx + 1] <= ts:
            h_idx += 1
        if h_idx >= len(ema_f) or pd.isna(ema_f.iloc[h_idx]) or pd.isna(ema_s.iloc[h_idx]):
            regime_map[i] = NONE_SIG
        elif ema_f.iloc[h_idx] > ema_s.iloc[h_idx]:
            regime_map[i] = BUY
        elif ema_f.iloc[h_idx] < ema_s.iloc[h_idx]:
            regime_map[i] = SELL
        else:
            regime_map[i] = NONE_SIG
    return regime_map


# ── ATR Expansion Filter ─────────────────────────────────────────

def atr_expanding(atr_s, idx, cfg=CONFIG):
    if not cfg['atr_expansion_filter']:
        return True
    if idx < 20:
        return True
    avg_atr = atr_s.iloc[idx - 20: idx].mean()
    if avg_atr == 0:
        return True
    return atr_s.iloc[idx] >= cfg['atr_expansion_ratio'] * avg_atr


# ── Adaptive Risk ─────────────────────────────────────────────────

def get_adaptive_risk_pct(trade_history: list, cfg=CONFIG) -> float:
    if not cfg['adaptive_risk'] or len(trade_history) < 10:
        return cfg['risk_pct']
    recent = trade_history[-cfg['adaptive_lookback']:]
    wins = sum(1 for t in recent if t.get('pnl', 0) > 0)
    wr = wins / len(recent)
    if wr > 0.50:
        return cfg['risk_pct_high']
    elif wr >= 0.40:
        return cfg['risk_pct_med']
    return cfg['risk_pct_low']


# ── Equity Curve Filter ──────────────────────────────────────────

def equity_below_ema(equity_history: list, capital: float, cfg=CONFIG) -> bool:
    if not cfg['equity_curve_filter'] or len(equity_history) < cfg['equity_ema_period']:
        return False
    eq = pd.Series(equity_history)
    ema_val = eq.ewm(span=cfg['equity_ema_period'], adjust=False).mean().iloc[-1]
    return capital < ema_val


# ── Trade Builder ─────────────────────────────────────────────────

def build_trade(sig, df, atr_s, idx, capital, trade_history=None,
                equity_history=None, cfg=CONFIG):
    """
    Build a trade dict with all fields needed for advanced exit management.
    Returns None if trade is invalid.
    """
    c = df.iloc[idx]
    entry = float(c['close'])
    atr_val = float(atr_s.iloc[idx])

    # Capped capital for risk sizing
    capped = min(capital, cfg['initial_capital'] * 3)
    capped = max(capped, cfg['initial_capital'] * 0.5)

    current_risk_pct = get_adaptive_risk_pct(trade_history or [], cfg)
    risk_usd = capped * current_risk_pct

    # Equity curve filter
    if equity_history and equity_below_ema(equity_history, capital, cfg):
        risk_usd *= 0.5

    if sig == BUY:
        sl = float(c['low']) - cfg['atr_sl_mult'] * atr_val
        if sl >= entry:
            return None
        rpu = entry - sl
    else:
        sl = float(c['high']) + cfg['atr_sl_mult'] * atr_val
        if sl <= entry:
            return None
        rpu = sl - entry

    if rpu < 0.05 * atr_val or rpu > cfg['max_rr_sl_atr'] * atr_val:
        return None

    target = entry + cfg['rr_ratio'] * rpu if sig == BUY else entry - cfg['rr_ratio'] * rpu

    return {
        'sig': sig,
        'signal': 'BUY' if sig == BUY else 'SELL',
        'entry_px': round(entry, 2),
        'sl': round(sl, 2),
        'initial_sl': round(sl, 2),
        'tgt': round(target, 2),
        'risk_usd': round(risk_usd, 2),
        'rpu': round(rpu, 2),
        'risk_pct_used': current_risk_pct,
        'remaining_pct': 1.0,
        'partial_pnl': 0.0,
        'breakeven_hit': False,
        'trailing_active': False,
        'candle_type': '',
    }


# ── Signal Checker (for forward test / bot) ──────────────────────

def precompute_indicators(df, sec_df, cfg=CONFIG):
    """Pre-compute all indicators for signal checking."""
    ema9 = ema(df['close'], cfg['ema_fast'])
    ema21 = ema(df['close'], cfg['ema_slow'])
    ema50 = ema(df['close'], cfg['ema_htf'])
    atr14 = atr(df, cfg['atr_period'])
    vol_sma = df['volume'].rolling(cfg['volume_period']).mean()

    pct_k, pct_d = (stoch(df, cfg['stoch_k'], cfg['stoch_d'])
                     if cfg['use_stoch'] else (None, None))

    s_ema9 = ema(sec_df['close'], cfg['ema_fast'])
    s_ema21 = ema(sec_df['close'], cfg['ema_slow'])
    s_ema50 = ema(sec_df['close'], cfg['ema_htf'])
    s_atr = atr(sec_df, cfg['atr_period'])

    htf_regime = compute_htf_regime(df, cfg)

    return {
        'ema9': ema9, 'ema21': ema21, 'ema50': ema50,
        'atr': atr14, 'vol_sma': vol_sma,
        'pct_k': pct_k, 'pct_d': pct_d,
        's_ema9': s_ema9, 's_ema21': s_ema21, 's_ema50': s_ema50, 's_atr': s_atr,
        'htf_regime': htf_regime,
    }


def check_signal(df, sec_df, indicators=None, cfg=CONFIG):
    """
    Check for entry signal on the last candle.
    Returns (direction, candle_type) or (None, None).
    """
    idx = len(df) - 1
    warmup = max(cfg['ema_htf'], cfg['atr_period'], cfg['volume_period'],
                 (cfg['stoch_k'] + cfg['stoch_d']) if cfg['use_stoch'] else 0) + 5

    if idx < warmup:
        return None, None

    # Time filter
    ts = df.iloc[idx]['datetime']
    h = ts.hour
    if h in cfg['blocked_hours']:
        return None, None

    # Pre-compute if not provided
    if indicators is None:
        indicators = precompute_indicators(df, sec_df, cfg)

    ind = indicators

    # Layer 1: Trend detection
    sig = detect_trend(df, ind['ema9'], ind['ema21'], ind['ema50'], ind['atr'], idx, cfg)
    if sig == NONE_SIG:
        return None, None

    # Layer 2: HTF regime filter
    if not cfg['allow_counter_trend']:
        htf_bias = ind['htf_regime'].get(idx)
        if htf_bias is not None and htf_bias != NONE_SIG and htf_bias != sig:
            return None, None

    # Layer 3: ATR expansion
    if not atr_expanding(ind['atr'], idx, cfg):
        return None, None

    # Layer 4: Entry candle pattern
    ctype = entry_candle(df, ind['ema9'], ind['ema21'], ind['atr'],
                         ind['vol_sma'], idx, sig, cfg)
    if ctype is None:
        return None, None

    # Layer 5: Stochastic confirmation
    if cfg['use_stoch'] and not stoch_confirms(ind['pct_k'], ind['pct_d'], idx, sig):
        return None, None

    # Layer 6: Correlation filter
    s_idx = min(idx, len(sec_df) - 1)
    if not corr_confirms(sig, sec_df, ind['s_ema9'], ind['s_ema21'],
                         ind['s_ema50'], ind['s_atr'], s_idx, cfg):
        return None, None

    return sig, ctype


# ── Exit Checker (advanced) ──────────────────────────────────────

def check_exit(active: dict, candle: pd.Series, candles_held: int,
               atr_val: float, cfg=CONFIG):
    """
    Advanced exit management with trailing stop, partial TP, time exit.
    Returns (exit_price, pnl, updated_active) or (None, None, active).
    Mutates active dict in place for SL updates.
    """
    hi, lo = float(candle['high']), float(candle['low'])
    close = float(candle['close'])
    is_buy = active['sig'] == BUY
    rpu = abs(active['entry_px'] - active['initial_sl'])

    if rpu == 0:
        return None, None, active

    # R-multiples at candle extremes
    if is_buy:
        best_r = (hi - active['entry_px']) / rpu
    else:
        best_r = (active['entry_px'] - lo) / rpu

    # 1. SL hit
    if is_buy and lo <= active['sl']:
        pnl = _calc_pnl(active, active['sl'])
        return float(active['sl']), pnl, active
    if not is_buy and hi >= active['sl']:
        pnl = _calc_pnl(active, active['sl'])
        return float(active['sl']), pnl, active

    # 2. Full TP hit
    if is_buy and hi >= active['tgt']:
        pnl = _calc_pnl(active, active['tgt'])
        return float(active['tgt']), pnl, active
    if not is_buy and lo <= active['tgt']:
        pnl = _calc_pnl(active, active['tgt'])
        return float(active['tgt']), pnl, active

    # 3. Time exit
    if candles_held >= cfg['max_hold_candles']:
        pnl = _calc_pnl(active, close)
        return close, pnl, active

    # 4. Partial TP at +1R
    if cfg['partial_tp'] and active['remaining_pct'] > (1.0 - cfg['partial_tp_pct'] + 0.01):
        partial_price = (active['entry_px'] + cfg['partial_tp_at_r'] * rpu if is_buy
                         else active['entry_px'] - cfg['partial_tp_at_r'] * rpu)
        if (is_buy and hi >= partial_price) or (not is_buy and lo <= partial_price):
            # Execute partial close
            closing_units = active['risk_usd'] / rpu * cfg['partial_tp_pct']
            if is_buy:
                active['partial_pnl'] += (partial_price - active['entry_px']) * closing_units
            else:
                active['partial_pnl'] += (active['entry_px'] - partial_price) * closing_units
            active['remaining_pct'] -= cfg['partial_tp_pct']

    # 5. Move SL to breakeven after +1R
    if cfg['trailing_stop'] and not active['breakeven_hit'] and best_r >= cfg['breakeven_at_r']:
        active['sl'] = active['entry_px']
        active['breakeven_hit'] = True

    # 6. Trailing stop after +1.5R
    if cfg['trailing_stop'] and active['breakeven_hit'] and best_r >= cfg['trail_after_r']:
        active['trailing_active'] = True
        trail_dist = cfg['trail_atr_mult'] * atr_val
        if is_buy:
            new_sl = hi - trail_dist
            active['sl'] = max(active['sl'], round(new_sl, 2))
        else:
            new_sl = lo + trail_dist
            active['sl'] = min(active['sl'], round(new_sl, 2))

    return None, None, active


def _calc_pnl(active: dict, exit_price: float) -> float:
    """Calculate PnL including partial profits."""
    rpu = abs(active['entry_px'] - active['initial_sl'])
    if rpu == 0:
        return 0.0
    units = active['risk_usd'] / rpu
    active_units = units * active['remaining_pct']
    if active['sig'] == BUY:
        pnl = (exit_price - active['entry_px']) * active_units
    else:
        pnl = (active['entry_px'] - exit_price) * active_units
    return round(pnl + active.get('partial_pnl', 0), 2)
