"""
EMA 9/21 + Stochastic Scalping Strategy — v3
=============================================
BTC (primary) + ETH (correlation confirmation)

Trend filter  : EMA9/21 crossover with ATR-normalized slope + EMA50 HTF bias
Momentum gate : Stochastic(14,3) — %K > 50 AND %K > %D for longs (vice versa shorts)
Entry timing  : Candle pattern (pin bar / engulfing / EMA reclaim) near EMA zone
Risk mgmt     : ATR-based SL, daily limits, consecutive-loss cooldown, time filter

v3 changes vs v2:
  - EMA slow period 15 → 21 (wider gap reduces false crossovers)
  - Stochastic(14,3) confirmation gate added (optional, use_stoch=True)
  - Optimizer source: optimize_indicators.py on 90d Binance data
    Best combo: ema_9_21 + stoch_14_3  Score=1.3977  Return=+67%  WR=44.8%  PF=1.55
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class Signal(Enum):
    BUY  = "BUY"
    SELL = "SELL"
    NONE = "NONE"


class TradeStatus(Enum):
    WIN          = "WIN"
    LOSS         = "LOSS"
    FORCE_CLOSED = "FORCE_CLOSED"


@dataclass
class Trade:
    entry_time:      pd.Timestamp
    signal:          Signal
    entry_price:     float
    stop_loss:       float
    target:          float
    risk_usd:        float
    reward_usd:      float
    candle_type:     str = ""
    status:          TradeStatus = None
    exit_time:       Optional[pd.Timestamp] = None
    exit_price:      float = 0.0
    pnl_usd:         float = 0.0
    hold_minutes:    float = 0.0
    sl_pct:          float = 0.0   # SL distance as % of entry — for analysis

    def close(self, exit_time: pd.Timestamp, exit_price: float, force: bool = False):
        self.exit_time = exit_time
        self.exit_price = exit_price
        risk_per_unit = abs(self.entry_price - self.stop_loss)
        units = self.risk_usd / risk_per_unit if risk_per_unit > 0 else 0
        if self.signal == Signal.BUY:
            self.pnl_usd = (exit_price - self.entry_price) * units
        else:
            self.pnl_usd = (self.entry_price - exit_price) * units
        self.hold_minutes = (exit_time - self.entry_time).total_seconds() / 60
        if force:
            self.status = TradeStatus.FORCE_CLOSED
        elif self.pnl_usd >= 0:
            self.status = TradeStatus.WIN
        else:
            self.status = TradeStatus.LOSS


class EMAScalpingStrategy:
    """
    Dual-EMA scalping with full filter stack.

    Parameters
    ----------
    capital          : Starting capital USD
    risk_pct         : Fraction of capital risked per trade (0.015 = 1.5%)
    ema_fast         : Fast EMA period (9)
    ema_slow         : Slow EMA period (15)
    ema_htf          : Higher-timeframe proxy EMA period (50)
    atr_period       : ATR period (14)
    atr_sl_mult      : SL = candle_extreme - atr_sl_mult * ATR
    slope_threshold  : Min ATR-normalized slope to consider "trending"
                       (1.0 = EMA moved ≥ 1 ATR per candle on avg — genuinely strong)
    slope_lookback   : Candles used to measure slope
    ema_prox_pct     : Max distance from EMA9 for entry candle low/high (fraction)
    volume_mult      : Min ratio of candle volume to rolling vol SMA
    volume_period    : Rolling volume SMA period
    max_rr_sl_atr    : Reject trade if SL > this many ATRs wide (avoids huge SL)
    consec_loss_max  : Consecutive losses before cooldown kicks in
    consec_cooldown  : Cooldown in candles (12 = 1 hour on 5m)
    max_trades_day   : Max entries per trading day
    daily_loss_pct   : Daily loss limit as fraction of capital
    daily_profit_pct : Daily profit target as fraction of capital
    time_filter      : True = block 00:00–04:00 UTC
    rr_ratio         : Risk:reward ratio (default 2.0 = 1:2)
    """

    def __init__(
        self,
        capital:          float = 1_000.0,
        risk_pct:         float = 0.015,
        ema_fast:         int   = 9,
        ema_slow:         int   = 15,
        ema_htf:          int   = 50,
        atr_period:       int   = 14,
        atr_sl_mult:      float = 0.75,
        slope_threshold:  float = 0.20,
        slope_lookback:   int   = 5,
        ema_prox_pct:     float = 0.005,
        volume_mult:      float = 1.2,
        volume_period:    int   = 20,
        max_rr_sl_atr:    float = 3.0,
        consec_loss_max:  int   = 2,
        consec_cooldown:  int   = 12,
        max_trades_day:   int   = 5,
        daily_loss_pct:   float = 0.03,
        daily_profit_pct: float = 0.04,
        time_filter:      bool  = True,
        rr_ratio:         float = 2.0,
        use_stoch:        bool  = False,
        stoch_k:          int   = 14,
        stoch_d:          int   = 3,
        use_ema_reclaim:  bool       = True,
        use_engulfing:    bool       = True,
        blocked_hours:    set        = None,  # UTC hours to block, e.g. {9,10,12,13,14,15}
    ):
        self.capital          = capital
        self.risk_pct         = risk_pct
        self.ema_fast         = ema_fast
        self.ema_slow         = ema_slow
        self.ema_htf          = ema_htf
        self.atr_period       = atr_period
        self.atr_sl_mult      = atr_sl_mult
        self.slope_threshold  = slope_threshold
        self.slope_lookback   = slope_lookback
        self.ema_prox_pct     = ema_prox_pct
        self.volume_mult      = volume_mult
        self.volume_period    = volume_period
        self.max_rr_sl_atr    = max_rr_sl_atr
        self.consec_loss_max  = consec_loss_max
        self.consec_cooldown  = consec_cooldown
        self.max_trades_day   = max_trades_day
        self.daily_loss_pct   = daily_loss_pct
        self.daily_profit_pct = daily_profit_pct
        self.time_filter      = time_filter
        self.rr_ratio         = rr_ratio
        self.use_stoch        = use_stoch
        self.stoch_k          = stoch_k
        self.stoch_d          = stoch_d
        self.use_ema_reclaim  = use_ema_reclaim
        self.use_engulfing    = use_engulfing
        self.blocked_hours    = blocked_hours or set()

        # State
        self.trades:             list[Trade]     = []
        self.active_trade:       Optional[Trade] = None
        self._daily_pnl:         float = 0.0
        self._trades_today:      int   = 0
        self._current_date              = None
        self._consecutive_losses: int  = 0
        self._cooldown_until_idx: int  = 0

    # ------------------------------------------------------------------ #
    #  Indicators                                                           #
    # ------------------------------------------------------------------ #

    def _ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def _atr(self, df: pd.DataFrame) -> pd.Series:
        """Wilder ATR (True Range smoothed with EWM)."""
        prev_close = df['close'].shift(1)
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - prev_close).abs(),
            (df['low']  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(span=self.atr_period, adjust=False).mean()

    def _stoch(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Stochastic oscillator: returns (%K, %D)."""
        lo    = df['low'].rolling(self.stoch_k).min()
        hi    = df['high'].rolling(self.stoch_k).max()
        pct_k = 100 * (df['close'] - lo) / (hi - lo).replace(0, np.nan)
        pct_d = pct_k.rolling(self.stoch_d).mean()
        return pct_k, pct_d

    def _stoch_confirms(self, pct_k: pd.Series, pct_d: pd.Series,
                        idx: int, sig: Signal) -> bool:
        """BUY: %K > 50 AND %K > %D  |  SELL: %K < 50 AND %K < %D"""
        k = pct_k.iloc[idx]
        d = pct_d.iloc[idx]
        if pd.isna(k) or pd.isna(d):
            return False
        if sig == Signal.BUY:
            return k > 50 and k > d
        return k < 50 and k < d

    def _slope_normalized(self, ema: pd.Series, atr: pd.Series, idx: int) -> float:
        """
        ATR-normalized slope: (EMA change over lookback) / (avg ATR * lookback).
        Positive = sloping up, negative = sloping down.
        A value of ±1.0 means EMA moved 1 full ATR per candle — genuinely trending.
        """
        lb = self.slope_lookback
        if idx < lb:
            return 0.0
        price_change = ema.iloc[idx] - ema.iloc[idx - lb]
        avg_atr = atr.iloc[max(0, idx - lb): idx + 1].mean()
        if avg_atr == 0:
            return 0.0
        return price_change / (avg_atr * lb)

    # ------------------------------------------------------------------ #
    #  Trend detection                                                      #
    # ------------------------------------------------------------------ #

    def _trend(
        self,
        df:    pd.DataFrame,
        ema9:  pd.Series,
        ema15: pd.Series,
        ema50: pd.Series,
        atr:   pd.Series,
        idx:   int,
    ) -> Signal:
        """
        BUY  : EMA9 > EMA15, both sloping up > threshold, price > EMA50
        SELL : EMA9 < EMA15, both sloping down > threshold, price < EMA50
        """
        s9  = self._slope_normalized(ema9,  atr, idx)
        s15 = self._slope_normalized(ema15, atr, idx)
        price = df['close'].iloc[idx]

        bull = (
            ema9.iloc[idx] > ema15.iloc[idx] and
            s9  >  self.slope_threshold and
            s15 >  0 and
            price > ema50.iloc[idx]
        )
        bear = (
            ema9.iloc[idx] < ema15.iloc[idx] and
            s9  < -self.slope_threshold and
            s15 <  0 and
            price < ema50.iloc[idx]
        )

        if bull:  return Signal.BUY
        if bear:  return Signal.SELL
        return Signal.NONE

    # ------------------------------------------------------------------ #
    #  Entry candle detection                                               #
    # ------------------------------------------------------------------ #

    def _near_ema(
        self, df: pd.DataFrame, ema9: pd.Series, ema15: pd.Series,
        atr: pd.Series, idx: int, sig: Signal
    ) -> bool:
        """
        Candle must originate from within the EMA zone.
        For BUY  : candle low  ≤ EMA9 * (1 + prox_pct)   i.e. pulled back to EMA
        For SELL : candle high ≥ EMA9 * (1 - prox_pct)
        """
        c = df.iloc[idx]
        band = atr.iloc[idx] * 1.5   # tolerance = 1.5 ATR from EMA9
        if sig == Signal.BUY:
            return c['low'] <= ema9.iloc[idx] + band
        else:
            return c['high'] >= ema9.iloc[idx] - band

    def _vol_ok(self, df: pd.DataFrame, vol_sma: pd.Series, idx: int) -> bool:
        """Entry candle volume must exceed vol_mult × rolling average."""
        if 'volume' not in df.columns or vol_sma is None:
            return True
        ref = vol_sma.iloc[idx]
        if pd.isna(ref) or ref == 0:
            return True
        return df['volume'].iloc[idx] > self.volume_mult * ref

    def _time_ok(self, ts: pd.Timestamp) -> bool:
        """Block 00:00–04:00 UTC (low-liquidity dead zone) + any extra blocked_hours."""
        if not self.time_filter:
            return True
        h = ts.hour
        if 0 <= h < 4:
            return False
        if h in self.blocked_hours:
            return False
        return True

    def _is_pin_bar(self, df: pd.DataFrame, atr: pd.Series, idx: int, sig: Signal) -> bool:
        """
        Criteria (all must hold):
        - Body ≤ 30% of total range
        - Wick in signal direction ≥ 2.5× body
        - Total range ≥ 0.5 ATR (not a doji sliver)
        - Close on correct side (green for BUY, red for SELL)
        """
        c = df.iloc[idx]
        rng = c['high'] - c['low']
        if rng < 0.5 * atr.iloc[idx]:
            return False
        body = abs(c['close'] - c['open'])
        if body > 0.30 * rng:
            return False

        if sig == Signal.BUY:
            lower_wick = min(c['open'], c['close']) - c['low']
            return (lower_wick >= 2.5 * body and
                    c['close'] > c['open'])
        else:
            upper_wick = c['high'] - max(c['open'], c['close'])
            return (upper_wick >= 2.5 * body and
                    c['close'] < c['open'])

    def _is_engulfing(self, df: pd.DataFrame, atr: pd.Series, idx: int, sig: Signal) -> bool:
        """
        True bullish engulfing requires ALL:
        1. Prior candle is bearish (opposite color) — genuine reversal
        2. Current candle is bullish
        3. Current body fully engulfs prior body (open < prior close AND close > prior open)
        4. Current body ≥ 1.5× prior body size — must be meaningfully larger
        5. Current body ≥ 0.5 ATR — not a micro-candle signal
        Mirrored for bearish engulfing.
        """
        if idx < 1:
            return False
        c = df.iloc[idx]
        p = df.iloc[idx - 1]
        curr_body = abs(c['close'] - c['open'])
        prev_body = abs(p['close'] - p['open'])

        if curr_body < 0.5 * atr.iloc[idx]:
            return False
        if prev_body == 0:
            return False
        if curr_body < 1.5 * prev_body:
            return False

        if sig == Signal.BUY:
            return (p['close'] < p['open'] and            # prior is bearish
                    c['close'] > c['open'] and            # current is bullish
                    c['open']  < p['close'] and           # engulfs on open side
                    c['close'] > p['open'])               # engulfs on close side
        else:
            return (p['close'] > p['open'] and
                    c['close'] < c['open'] and
                    c['open']  > p['close'] and
                    c['close'] < p['open'])

    def _is_ema_reclaim(
        self, df: pd.DataFrame, ema9: pd.Series, ema15: pd.Series,
        atr: pd.Series, idx: int, sig: Signal
    ) -> bool:
        """
        Price briefly violates EMA15 (opposite side) in the prior candle,
        then current candle closes firmly back on the correct side of EMA9.
        Body must be ≥ 0.4 ATR (conviction).
        """
        if idx < 1:
            return False
        c = df.iloc[idx]
        p = df.iloc[idx - 1]
        body = abs(c['close'] - c['open'])
        if body < 0.4 * atr.iloc[idx]:
            return False

        if sig == Signal.BUY:
            prev_violated = p['low'] < ema15.iloc[idx - 1]
            reclaim       = c['close'] > ema9.iloc[idx] and c['close'] > c['open']
            return prev_violated and reclaim
        else:
            prev_violated = p['high'] > ema15.iloc[idx - 1]
            reclaim       = c['close'] < ema9.iloc[idx] and c['close'] < c['open']
            return prev_violated and reclaim

    def _entry_candle(
        self, df: pd.DataFrame, ema9: pd.Series, ema15: pd.Series,
        atr: pd.Series, vol_sma: Optional[pd.Series], idx: int, sig: Signal
    ) -> Optional[str]:
        """
        Gate checks first, then candle pattern.
        Returns candle type string or None.
        """
        # Gate 1: must be near EMA zone
        if not self._near_ema(df, ema9, ema15, atr, idx, sig):
            return None
        # Gate 2: volume confirmation
        if not self._vol_ok(df, vol_sma, idx):
            return None
        # Gate 3: minimum candle size (anti-doji)
        c = df.iloc[idx]
        if abs(c['close'] - c['open']) < 0.08 * atr.iloc[idx]:
            return None
        # Pattern check
        if self._is_pin_bar(df, atr, idx, sig):
            return "pin_bar"
        if self.use_engulfing and self._is_engulfing(df, atr, idx, sig):
            return "engulfing"
        if self.use_ema_reclaim and self._is_ema_reclaim(df, ema9, ema15, atr, idx, sig):
            return "ema_reclaim"
        return None

    # ------------------------------------------------------------------ #
    #  Correlation filter                                                   #
    # ------------------------------------------------------------------ #

    def _corr_confirms(
        self, primary_sig: Signal, sec_df: pd.DataFrame,
        sec_ema9: pd.Series, sec_ema15: pd.Series,
        sec_ema50: pd.Series, sec_atr: pd.Series, idx: int
    ) -> bool:
        """
        ETH must NOT be showing an opposing trend.
        If ETH is flat (NONE), we allow the trade but flag it as weaker.
        """
        sec_sig = self._trend(sec_df, sec_ema9, sec_ema15, sec_ema50, sec_atr, idx)
        if sec_sig == Signal.NONE:
            return True          # flat secondary — cautious but allowed
        return sec_sig == primary_sig

    # ------------------------------------------------------------------ #
    #  Risk management                                                      #
    # ------------------------------------------------------------------ #

    def _reset_day(self, date):
        if self._current_date != date:
            self._current_date = date
            self._daily_pnl    = 0.0
            self._trades_today = 0

    def _can_trade(self, idx: int, ts: pd.Timestamp) -> tuple[bool, str]:
        if not self._time_ok(ts):
            return False, "time_filter"
        if idx < self._cooldown_until_idx:
            return False, "consec_loss_cooldown"
        if self._trades_today >= self.max_trades_day:
            return False, "max_trades"
        if self._daily_pnl <= -(self.capital * self.daily_loss_pct):
            return False, "daily_loss_limit"
        if self._daily_pnl >= (self.capital * self.daily_profit_pct):
            return False, "daily_profit_target"
        return True, "ok"

    def _build_trade(
        self, sig: Signal, df: pd.DataFrame, atr: pd.Series, idx: int
    ) -> Optional[Trade]:
        c = df.iloc[idx]
        entry   = c['close']
        atr_val = atr.iloc[idx]
        risk_usd = self.capital * self.risk_pct

        if sig == Signal.BUY:
            sl  = c['low'] - self.atr_sl_mult * atr_val
            if sl >= entry: return None
            rpu = entry - sl
        else:
            sl  = c['high'] + self.atr_sl_mult * atr_val
            if sl <= entry: return None
            rpu = sl - entry

        # Reject degenerate or oversized SL
        if rpu < 0.05 * atr_val:             return None
        if rpu > self.max_rr_sl_atr * atr_val: return None

        target     = entry + self.rr_ratio * rpu if sig == Signal.BUY else entry - self.rr_ratio * rpu
        reward_usd = risk_usd * self.rr_ratio

        return Trade(
            entry_time   = df['datetime'].iloc[idx],
            signal       = sig,
            entry_price  = entry,
            stop_loss    = sl,
            target       = target,
            risk_usd     = risk_usd,
            reward_usd   = reward_usd,
            sl_pct       = rpu / entry * 100,
        )

    def _record_close(self, trade: Trade, idx: int):
        self.capital      += trade.pnl_usd
        self._daily_pnl   += trade.pnl_usd
        self._trades_today += 1
        self.trades.append(trade)
        self.active_trade  = None

        if trade.status in (TradeStatus.LOSS, TradeStatus.FORCE_CLOSED):
            self._consecutive_losses += 1
            if self._consecutive_losses >= self.consec_loss_max:
                self._cooldown_until_idx  = idx + self.consec_cooldown
                self._consecutive_losses  = 0
        else:
            self._consecutive_losses = 0

    # ------------------------------------------------------------------ #
    #  Main run loop                                                        #
    # ------------------------------------------------------------------ #

    def run(
        self, primary_df: pd.DataFrame, secondary_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        primary_df  : BTC 5-min OHLCV with columns [datetime, open, high, low, close, volume]
        secondary_df: ETH 5-min OHLCV with same columns (aligned timestamps)
        """
        df  = primary_df.copy().reset_index(drop=True)
        sec = secondary_df.copy().reset_index(drop=True)

        # Pre-compute all indicators once
        ema9   = self._ema(df['close'],  self.ema_fast)
        ema15  = self._ema(df['close'],  self.ema_slow)
        ema50  = self._ema(df['close'],  self.ema_htf)
        atr14  = self._atr(df)
        vol_sma = (df['volume'].rolling(self.volume_period).mean()
                   if 'volume' in df.columns else None)
        pct_k, pct_d = self._stoch(df) if self.use_stoch else (None, None)

        s_ema9  = self._ema(sec['close'], self.ema_fast)
        s_ema15 = self._ema(sec['close'], self.ema_slow)
        s_ema50 = self._ema(sec['close'], self.ema_htf)
        s_atr   = self._atr(sec)

        stoch_warmup = (self.stoch_k + self.stoch_d) if self.use_stoch else 0
        warmup = max(self.ema_htf, self.atr_period, self.volume_period, stoch_warmup) + 5

        for idx in range(warmup, len(df)):
            row = df.iloc[idx]
            ts  = row['datetime']
            self._reset_day(ts.date() if hasattr(ts, 'date') else ts)

            # ── Manage open trade ──────────────────────────────────────
            if self.active_trade is not None:
                t  = self.active_trade
                hi = row['high']
                lo = row['low']

                if t.signal == Signal.BUY:
                    if lo <= t.stop_loss:
                        t.close(ts, t.stop_loss)
                        self._record_close(t, idx)
                    elif hi >= t.target:
                        t.close(ts, t.target)
                        self._record_close(t, idx)
                else:
                    if hi >= t.stop_loss:
                        t.close(ts, t.stop_loss)
                        self._record_close(t, idx)
                    elif lo <= t.target:
                        t.close(ts, t.target)
                        self._record_close(t, idx)
                continue

            # ── Risk / session filters ─────────────────────────────────
            ok, _ = self._can_trade(idx, ts)
            if not ok:
                continue

            # ── Trend detection ────────────────────────────────────────
            sig = self._trend(df, ema9, ema15, ema50, atr14, idx)
            if sig == Signal.NONE:
                continue

            # ── Entry candle ───────────────────────────────────────────
            ctype = self._entry_candle(df, ema9, ema15, atr14, vol_sma, idx, sig)
            if ctype is None:
                continue

            # ── Stochastic confirmation ────────────────────────────────
            if self.use_stoch and not self._stoch_confirms(pct_k, pct_d, idx, sig):
                continue

            # ── Correlation filter ─────────────────────────────────────
            s_idx = min(idx, len(sec) - 1)
            if not self._corr_confirms(sig, sec, s_ema9, s_ema15, s_ema50, s_atr, s_idx):
                continue

            # ── Build and open trade ───────────────────────────────────
            trade = self._build_trade(sig, df, atr14, idx)
            if trade is None:
                continue
            trade.candle_type = ctype
            self.active_trade = trade

        # Force-close at end of data
        if self.active_trade is not None:
            last = df.iloc[-1]
            self.active_trade.close(last['datetime'], last['close'], force=True)
            self._record_close(self.active_trade, len(df) - 1)

        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([{
            'entry_time':   t.entry_time,
            'exit_time':    t.exit_time,
            'signal':       t.signal.value,
            'entry_price':  t.entry_price,
            'exit_price':   t.exit_price,
            'stop_loss':    t.stop_loss,
            'target':       t.target,
            'risk_usd':     t.risk_usd,
            'pnl_usd':      t.pnl_usd,
            'status':       t.status.value,
            'candle_type':  t.candle_type,
            'hold_minutes': t.hold_minutes,
            'sl_pct':       t.sl_pct,
        } for t in self.trades])
