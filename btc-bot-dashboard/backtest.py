"""
Comprehensive Backtest — BTC/ETH EMA 9/21 + Stochastic Scalping
================================================================
Run: python backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import ccxt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

from strategy import EMAScalpingStrategy

INITIAL_CAPITAL = 1_000.0
EXCHANGE_ID     = "binance"   # change to "bybit", "okx", etc. if preferred
BATCH_DAYS      = 3           # days per ccxt fetch call (3d × 288 candles/d = 864 < 1000)

_exchange: ccxt.Exchange | None = None

def get_exchange() -> ccxt.Exchange:
    global _exchange
    if _exchange is None:
        ex_cls   = getattr(ccxt, EXCHANGE_ID)
        _exchange = ex_cls({'enableRateLimit': True})
    return _exchange

# ─────────────────────────────────────────────────────────────── #
#  Data                                                            #
# ─────────────────────────────────────────────────────────────── #

def fetch(symbol: str, days: int = 60, interval: str = "5m") -> pd.DataFrame:
    """
    Fetch `days` of 5m OHLCV from Binance via ccxt.
    Splits into batches of BATCH_DAYS to stay under the 1000-candle limit.
    """
    print(f"  Fetching {symbol} ({days}d)...", end=" ", flush=True)
    ex = get_exchange()

    candles_per_day  = 24 * 60 // 5          # 288 for 5m
    batch_ms         = BATCH_DAYS * 24 * 3_600_000
    since_ms         = int(time.time() * 1000) - days * 24 * 3_600_000
    limit_per_batch  = BATCH_DAYS * candles_per_day + 1   # slight buffer

    all_rows: list[list] = []
    cursor = since_ms

    while cursor < int(time.time() * 1000):
        batch = ex.fetch_ohlcv(symbol, timeframe=interval,
                               since=cursor, limit=limit_per_batch)
        if not batch:
            break
        all_rows.extend(batch)
        cursor = batch[-1][0] + 1   # advance past last candle timestamp

    if not all_rows:
        raise ValueError(f"No data returned for {symbol}")

    df = pd.DataFrame(all_rows, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True).dt.tz_localize(None)
    df = (df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
          .drop_duplicates('datetime')
          .sort_values('datetime')
          .dropna()
          .reset_index(drop=True))

    print(f"{len(df)} candles | {df['datetime'].iloc[0].date()} → {df['datetime'].iloc[-1].date()}")
    return df


def align(primary: pd.DataFrame, secondary: pd.DataFrame):
    """Inner-join on datetime, carry volume for primary."""
    merged = pd.merge(
        primary,
        secondary[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(
            columns={'open': 's_open', 'high': 's_high', 'low': 's_low',
                     'close': 's_close', 'volume': 's_volume'}),
        on='datetime', how='inner'
    )
    pri = merged[['datetime', 'open', 'high', 'low', 'close', 'volume']].copy()
    sec = merged[['datetime']].copy()
    sec['open']   = merged['s_open']
    sec['high']   = merged['s_high']
    sec['low']    = merged['s_low']
    sec['close']  = merged['s_close']
    sec['volume'] = merged['s_volume']
    return pri, sec

# ─────────────────────────────────────────────────────────────── #
#  Metrics                                                         #
# ─────────────────────────────────────────────────────────────── #

def _consec(mask: pd.Series) -> int:
    """Max consecutive True values in a boolean series."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def compute_metrics(trades_df: pd.DataFrame, initial_capital: float) -> dict | None:
    if trades_df is None or trades_df.empty:
        return None

    tdf = trades_df.copy()
    n   = len(tdf)

    wins   = tdf[tdf['pnl_usd'] > 0]
    losses = tdf[tdf['pnl_usd'] <= 0]

    win_rate  = len(wins) / n * 100
    total_pnl = tdf['pnl_usd'].sum()
    avg_win   = wins['pnl_usd'].mean()   if len(wins)   else 0.0
    avg_loss  = losses['pnl_usd'].mean() if len(losses) else 0.0
    rr        = abs(avg_win / avg_loss)  if avg_loss != 0 else float('inf')

    gross_profit = wins['pnl_usd'].sum()
    gross_loss   = abs(losses['pnl_usd'].sum()) if len(losses) else 1e-9
    pf           = gross_profit / gross_loss

    # Equity curve & drawdown
    tdf['cum_pnl'] = tdf['pnl_usd'].cumsum()
    tdf['equity']  = initial_capital + tdf['cum_pnl']
    peak           = tdf['equity'].cummax()
    tdf['dd_pct']  = (tdf['equity'] - peak) / peak * 100
    max_dd         = tdf['dd_pct'].min()

    # Expectancy
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    # Sharpe (daily PnL series)
    tdf['date'] = tdf['entry_time'].dt.date
    daily = tdf.groupby('date')['pnl_usd'].sum()
    n_days = len(daily)
    sharpe = (daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else 0.0

    # Sortino (downside deviation only)
    neg = daily[daily < 0]
    down_std = neg.std() if len(neg) > 1 else 1e-9
    sortino = daily.mean() / down_std * np.sqrt(365)

    # Calmar
    ann_return = (tdf['equity'].iloc[-1] / initial_capital - 1) * (365 / max(n_days, 1)) * 100
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

    # Recovery factor
    recovery = total_pnl / abs(initial_capital * max_dd / 100) if max_dd != 0 else float('inf')

    # Consecutive streaks
    is_win = tdf['pnl_usd'] > 0
    max_c_wins   = _consec(is_win)
    max_c_losses = _consec(~is_win)

    # Trade frequency
    n_active_days = tdf['entry_time'].dt.date.nunique()
    trades_per_day = n / n_active_days if n_active_days else 0

    # Avg hold
    avg_hold = tdf['hold_minutes'].mean() if 'hold_minutes' in tdf.columns else 0.0

    # Hour-of-day breakdown
    tdf['hour'] = tdf['entry_time'].dt.hour
    def win_rate_fn(x): return (x > 0).mean() * 100
    hourly = (tdf.groupby('hour')['pnl_usd']
              .agg(count='count', total='sum', mean='mean',
                   win_rate=win_rate_fn))

    # Candle type breakdown
    def candle_wr(x): return (x > 0).mean() * 100
    candle = (tdf.groupby('candle_type')['pnl_usd']
              .agg(count='count', total='sum', mean='mean', win_rate=candle_wr))

    # Weekly PnL
    tdf['week'] = tdf['entry_time'].dt.to_period('W')
    weekly = tdf.groupby('week')['pnl_usd'].sum()

    # Signal direction breakdown
    signal_br = (tdf.groupby('signal')['pnl_usd']
                 .agg(count='count', total='sum', win_rate=win_rate_fn))

    return {
        'n':              n,
        'wins':           len(wins),
        'losses':         len(losses),
        'win_rate':       win_rate,
        'total_pnl':      total_pnl,
        'return_pct':     total_pnl / initial_capital * 100,
        'ann_return':     ann_return,
        'avg_win':        avg_win,
        'avg_loss':       avg_loss,
        'rr_ratio':       rr,
        'profit_factor':  pf,
        'expectancy':     expectancy,
        'max_dd_pct':     max_dd,
        'sharpe':         sharpe,
        'sortino':        sortino,
        'calmar':         calmar,
        'recovery':       recovery,
        'max_c_wins':     max_c_wins,
        'max_c_losses':   max_c_losses,
        'trades_per_day': trades_per_day,
        'avg_hold_min':   avg_hold,
        'best_trade':     tdf['pnl_usd'].max(),
        'worst_trade':    tdf['pnl_usd'].min(),
        'final_capital':  initial_capital + total_pnl,
        'hourly':         hourly,
        'candle':         candle,
        'weekly':         weekly,
        'signal_br':      signal_br,
        'tdf':            tdf,
    }

# ─────────────────────────────────────────────────────────────── #
#  Console report                                                  #
# ─────────────────────────────────────────────────────────────── #

def print_report(m: dict, ic: float):
    if m is None:
        print("  No trades generated."); return

    W = "═" * 56
    D = "─" * 56
    print(f"\n{W}")
    print(f"  BTC/ETH EMA 9/21 + STOCHASTIC — BACKTEST REPORT")
    print(f"{W}")
    print(f"  Capital        : ${ic:>10,.2f}  →  ${m['final_capital']:>10,.2f}")
    print(f"  Total Return   : {m['return_pct']:>+10.2f}%   "
          f"(ann. {m['ann_return']:+.1f}%)")
    print(f"  Total PnL      : ${m['total_pnl']:>+10,.2f}")
    print(f"{D}")
    print(f"  Trades         : {m['n']:>4}   "
          f"({m['trades_per_day']:.1f}/day on active days)")
    print(f"  Wins / Losses  : {m['wins']:>3} / {m['losses']:<3}")
    print(f"  Win Rate       : {m['win_rate']:>6.1f}%")
    print(f"  Avg Win        : ${m['avg_win']:>+8,.2f}")
    print(f"  Avg Loss       : ${m['avg_loss']:>+8,.2f}")
    print(f"  Reward:Risk    : {m['rr_ratio']:>6.2f}x")
    print(f"  Profit Factor  : {m['profit_factor']:>6.2f}")
    print(f"  Expectancy     : ${m['expectancy']:>+8,.2f}  per trade")
    print(f"{D}")
    print(f"  Max Drawdown   : {m['max_dd_pct']:>7.2f}%")
    print(f"  Sharpe Ratio   : {m['sharpe']:>7.2f}   (annualized)")
    print(f"  Sortino Ratio  : {m['sortino']:>7.2f}")
    print(f"  Calmar Ratio   : {m['calmar']:>7.2f}")
    print(f"  Recovery Factor: {m['recovery']:>7.2f}x")
    print(f"{D}")
    print(f"  Max Consec Wins  : {m['max_c_wins']}")
    print(f"  Max Consec Loss  : {m['max_c_losses']}")
    print(f"  Avg Hold Time    : {m['avg_hold_min']:.1f} min")
    print(f"  Best Trade       : ${m['best_trade']:>+,.2f}")
    print(f"  Worst Trade      : ${m['worst_trade']:>+,.2f}")
    print(f"{D}")

    # Candle breakdown
    print(f"  Entry Candle Breakdown:")
    cb = m['candle']
    for ct, row in cb.iterrows():
        bar = "█" * int(row['win_rate'] / 5)
        print(f"    {ct:<14} n={int(row['count']):>3}  "
              f"wr={row['win_rate']:>5.1f}%  {bar}  "
              f"avg=${row['mean']:>+6.2f}  total=${row['total']:>+7.2f}")

    # Signal breakdown
    print(f"\n  Trade Direction:")
    for sig, row in m['signal_br'].iterrows():
        print(f"    {sig:<5} n={int(row['count']):>3}  "
              f"wr={row['win_rate']:>5.1f}%  total=${row['total']:>+7.2f}")

    # Best hours
    best_hrs = m['hourly'].sort_values('total', ascending=False).head(5)
    print(f"\n  Best Trading Hours (UTC):")
    for hr, row in best_hrs.iterrows():
        print(f"    {hr:02d}:00  n={int(row['count']):>2}  "
              f"wr={row['win_rate']:>5.1f}%  pnl=${row['total']:>+7.2f}")

    # Last 8 trades
    print(f"\n  Last 8 Trades:")
    for _, t in m['tdf'].tail(8).iterrows():
        icon = "✓" if t['pnl_usd'] > 0 else "✗"
        print(f"    {icon} {t['entry_time'].strftime('%m/%d %H:%M')} "
              f"{t['signal']:<5} {t['candle_type']:<14} "
              f"entry={t['entry_price']:>9.2f} "
              f"exit={t['exit_price']:>9.2f}  "
              f"pnl=${t['pnl_usd']:>+7.2f}")
    print(f"{W}\n")

# ─────────────────────────────────────────────────────────────── #
#  Charts                                                          #
# ─────────────────────────────────────────────────────────────── #

def plot_dashboard(m: dict, ic: float, save_path: str = "backtest_results.png"):
    if m is None:
        print("Nothing to plot."); return

    tdf = m['tdf']
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.38)

    DARK  = '#0d1117'
    GRID  = '#21262d'
    GREEN = '#3fb950'
    RED   = '#f85149'
    BLUE  = '#58a6ff'
    GOLD  = '#e3b341'
    TEXT  = '#c9d1d9'
    MID   = '#8b949e'

    def style(ax, title=""):
        ax.set_facecolor(DARK)
        ax.tick_params(colors=TEXT, labelsize=7.5)
        ax.spines[:].set_color(GRID)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        if title: ax.set_title(title, color=TEXT, fontsize=9, pad=4)
        ax.grid(color=GRID, linewidth=0.5, alpha=0.7)

    # ── Row 0: Equity curve (full width) ─────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    eq  = tdf['equity'].values
    x   = range(len(eq))
    ax0.plot(x, eq, color=BLUE, linewidth=1.4, label='Equity')
    ax0.fill_between(x, eq, ic, where=np.array(eq) >= ic,
                     alpha=0.15, color=GREEN)
    ax0.fill_between(x, eq, ic, where=np.array(eq) < ic,
                     alpha=0.15, color=RED)
    ax0.axhline(ic, color=MID, linestyle='--', linewidth=0.8)
    # Trade markers
    for _, t in tdf.iterrows():
        c = GREEN if t['pnl_usd'] > 0 else RED
        ax0.axvline(t.name, color=c, alpha=0.08, linewidth=0.7)
    ax0.set_ylabel("Portfolio ($)", color=TEXT, fontsize=8)
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    summary = (f"Return: {m['return_pct']:+.1f}%  |  "
               f"Sharpe: {m['sharpe']:.2f}  |  "
               f"Win Rate: {m['win_rate']:.1f}%  |  "
               f"PF: {m['profit_factor']:.2f}  |  "
               f"MaxDD: {m['max_dd_pct']:.1f}%")
    style(ax0, f"Equity Curve — {summary}")

    # ── Row 1: Underwater / Drawdown chart (full width) ──────────
    ax1 = fig.add_subplot(gs[1, :])
    dd  = tdf['dd_pct'].values
    ax1.fill_between(range(len(dd)), dd, 0, color=RED, alpha=0.55)
    ax1.plot(range(len(dd)), dd, color=RED, linewidth=0.8)
    ax1.axhline(m['max_dd_pct'], color=GOLD, linestyle='--',
                linewidth=0.9, label=f"Max DD {m['max_dd_pct']:.1f}%")
    ax1.axhline(0, color=MID, linewidth=0.5)
    ax1.legend(fontsize=7, labelcolor=TEXT, facecolor=DARK)
    ax1.set_ylabel("Drawdown (%)", color=TEXT, fontsize=8)
    ax1.invert_yaxis()
    style(ax1, "Underwater Chart (Drawdown)")

    # ── Row 2, Col 0: Per-trade PnL bars ─────────────────────────
    ax2 = fig.add_subplot(gs[2, 0])
    clrs = [GREEN if p > 0 else RED for p in tdf['pnl_usd']]
    ax2.bar(range(len(tdf)), tdf['pnl_usd'], color=clrs, alpha=0.85, width=0.85)
    ax2.axhline(0, color=MID, linewidth=0.7)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax2.set_xlabel("Trade #", color=TEXT, fontsize=7)
    win_p = mpatches.Patch(color=GREEN, label=f"Win ({m['wins']})")
    los_p = mpatches.Patch(color=RED,   label=f"Loss ({m['losses']})")
    ax2.legend(handles=[win_p, los_p], fontsize=7, facecolor=DARK, labelcolor=TEXT)
    style(ax2, "Per-Trade PnL")

    # ── Row 2, Col 1: Rolling 15-trade win rate ───────────────────
    ax3 = fig.add_subplot(gs[2, 1])
    roll_wr = (tdf['pnl_usd'] > 0).rolling(15, min_periods=5).mean() * 100
    ax3.plot(range(len(roll_wr)), roll_wr, color=BLUE, linewidth=1.2)
    ax3.axhline(50, color=MID,  linestyle='--', linewidth=0.8, label='50%')
    ax3.axhline(m['win_rate'], color=GOLD, linestyle=':', linewidth=0.9,
                label=f"Overall {m['win_rate']:.1f}%")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("Win Rate (%)", color=TEXT, fontsize=8)
    ax3.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)
    style(ax3, "Rolling Win Rate (15 trades)")

    # ── Row 2, Col 2: Hold time histogram ────────────────────────
    ax4 = fig.add_subplot(gs[2, 2])
    if 'hold_minutes' in tdf.columns:
        win_hold  = tdf.loc[tdf['pnl_usd'] > 0,  'hold_minutes']
        loss_hold = tdf.loc[tdf['pnl_usd'] <= 0, 'hold_minutes']
        bins = np.linspace(0, tdf['hold_minutes'].quantile(0.97), 25)
        ax4.hist(win_hold,  bins=bins, color=GREEN, alpha=0.6, label='Wins')
        ax4.hist(loss_hold, bins=bins, color=RED,   alpha=0.6, label='Losses')
        ax4.set_xlabel("Hold Duration (min)", color=TEXT, fontsize=7)
        ax4.legend(fontsize=7, facecolor=DARK, labelcolor=TEXT)
    style(ax4, "Trade Duration Distribution")

    # ── Row 3, Col 0: Hour-of-day win rate bar ────────────────────
    ax5 = fig.add_subplot(gs[3, 0])
    hr  = m['hourly']
    bar_c = [GREEN if wr >= 50 else RED for wr in hr['win_rate']]
    bars = ax5.bar(hr.index, hr['win_rate'], color=bar_c, alpha=0.8, width=0.7)
    ax5.axhline(50, color=MID, linestyle='--', linewidth=0.8)
    for bar, (idx_, row_) in zip(bars, hr.iterrows()):
        if row_['count'] > 0:
            ax5.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f"n={int(row_['count'])}", ha='center',
                     va='bottom', fontsize=5.5, color=MID)
    ax5.set_xlabel("Hour (UTC)", color=TEXT, fontsize=7)
    ax5.set_ylabel("Win Rate (%)", color=TEXT, fontsize=8)
    ax5.set_ylim(0, 105)
    style(ax5, "Win Rate by Hour (UTC)")

    # ── Row 3, Col 1: Weekly PnL bars ────────────────────────────
    ax6 = fig.add_subplot(gs[3, 1])
    wk  = m['weekly']
    wk_c = [GREEN if v > 0 else RED for v in wk.values]
    ax6.bar(range(len(wk)), wk.values, color=wk_c, alpha=0.85, width=0.7)
    ax6.axhline(0, color=MID, linewidth=0.7)
    ax6.set_xticks(range(len(wk)))
    ax6.set_xticklabels([str(p)[-5:] for p in wk.index],
                        rotation=40, ha='right', fontsize=6)
    ax6.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax6.set_ylabel("PnL ($)", color=TEXT, fontsize=8)
    style(ax6, "Weekly PnL")

    # ── Row 3, Col 2: Candle type quality ────────────────────────
    ax7 = fig.add_subplot(gs[3, 2])
    cb   = m['candle']
    xi   = np.arange(len(cb))
    w    = 0.35
    bars_wr  = ax7.bar(xi - w/2, cb['win_rate'], width=w,
                       color=BLUE,  alpha=0.8, label='Win Rate (%)')
    bars_avg = ax7.twinx()
    bars_avg.bar(xi + w/2, cb['mean'], width=w,
                 color=GOLD, alpha=0.8, label='Avg PnL ($)')
    bars_avg.set_ylabel("Avg PnL ($)", color=GOLD, fontsize=7)
    bars_avg.tick_params(colors=GOLD, labelsize=7)
    bars_avg.spines[:].set_color(GRID)
    ax7.axhline(50, color=MID, linestyle='--', linewidth=0.8)
    ax7.set_xticks(xi)
    ax7.set_xticklabels(cb.index, fontsize=7, color=TEXT)
    ax7.set_ylabel("Win Rate (%)", color=BLUE, fontsize=8)
    ax7.set_ylim(0, 105)
    h1, l1 = ax7.get_legend_handles_labels()
    h2, l2 = bars_avg.get_legend_handles_labels()
    ax7.legend(h1 + h2, l1 + l2, fontsize=6.5, facecolor=DARK, labelcolor=TEXT)
    style(ax7, "Entry Candle Quality")

    fig.suptitle("BTC/ETH  ·  EMA 9/21 + Stochastic  ·  $1,000 Capital  ·  5-min Chart",
                 color=TEXT, fontsize=13, fontweight='bold', y=1.005)

    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  Chart saved → {save_path}")


def plot_price_chart(df: pd.DataFrame, trades_df: pd.DataFrame,
                     strategy: EMAScalpingStrategy, last_n_days: int = 7,
                     save_path: str = "price_chart.png"):
    """
    Last N days of 5m price action with EMA overlays and trade entries/exits.
    """
    cutoff = df['datetime'].iloc[-1] - pd.Timedelta(days=last_n_days)
    sub    = df[df['datetime'] >= cutoff].copy().reset_index(drop=True)
    t_sub  = trades_df[trades_df['entry_time'] >= cutoff] if not trades_df.empty else pd.DataFrame()

    ema9  = strategy._ema(sub['close'], strategy.ema_fast)
    ema15 = strategy._ema(sub['close'], strategy.ema_slow)
    ema50 = strategy._ema(sub['close'], strategy.ema_htf)

    DARK  = '#0d1117'
    TEXT  = '#c9d1d9'
    GRID  = '#21262d'
    GREEN = '#3fb950'
    RED   = '#f85149'

    fig, ax = plt.subplots(figsize=(16, 6), facecolor=DARK)
    ax.set_facecolor(DARK)

    ax.plot(sub.index, sub['close'], color='#58a6ff', linewidth=0.8, label='BTC Close', zorder=2)
    ax.plot(sub.index, ema9,  color='#f0883e', linewidth=1.0, label='EMA 9',  zorder=3)
    ax.plot(sub.index, ema15, color='#a371f7', linewidth=1.0, label='EMA 15', zorder=3)
    ax.plot(sub.index, ema50, color='#e3b341', linewidth=0.8, linestyle='--',
            label='EMA 50', alpha=0.6, zorder=3)

    # Trade markers
    for _, t in t_sub.iterrows():
        ei = sub[sub['datetime'] == t['entry_time']].index
        xi = sub[sub['datetime'] == t['exit_time']].index
        color = GREEN if t['pnl_usd'] > 0 else RED
        if len(ei):
            marker = '^' if t['signal'] == 'BUY' else 'v'
            ax.scatter(ei[0], t['entry_price'], color=color,
                       marker=marker, s=70, zorder=5)
            ax.axhline(t['stop_loss'], color=RED,   alpha=0.2,
                       linewidth=0.6, linestyle=':')
            ax.axhline(t['target'],   color=GREEN,  alpha=0.2,
                       linewidth=0.6, linestyle=':')
        if len(xi):
            ax.scatter(xi[0], t['exit_price'], color=color,
                       marker='o', s=40, zorder=5, alpha=0.7)

    # x-axis labels every 2 hours
    tick_step = 24   # 24 × 5min = 120 min = 2 hours
    ticks = sub.index[::tick_step]
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [sub['datetime'].iloc[i].strftime('%m/%d %H:%M') for i in ticks],
        rotation=35, ha='right', fontsize=6.5, color=TEXT
    )
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(GRID)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_ylabel("BTC Price ($)", color=TEXT, fontsize=9)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.6)
    ax.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT, loc='upper left')
    ax.set_title(f"BTC/USD — Last {last_n_days} days (5m)  |  "
                 f"▲=Buy entry  ▼=Sell entry  ●=Exit",
                 color=TEXT, fontsize=10, pad=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=DARK)
    plt.close()
    print(f"  Chart saved → {save_path}")


# ─────────────────────────────────────────────────────────────── #
#  Main                                                            #
# ─────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60,
                        help="Number of days to backtest (default: 60)")
    args = parser.parse_args()

    print("\n" + "═" * 56)
    print("  BTC/ETH  EMA 9/21 + Stochastic  —  $1,000 Backtest")
    print("═" * 56)
    print("\nDownloading data...")

    btc_raw = fetch("BTC/USDT", days=args.days, interval="5m")
    eth_raw = fetch("ETH/USDT", days=args.days, interval="5m")
    btc, eth = align(btc_raw, eth_raw)
    print(f"  Aligned: {len(btc)} candles\n")

    # ── Optimizer-validated parameters ─────────────────────────────
    # Source: optimize_indicators.py — 90d Binance data (6,192 combos tested)
    #   Best combo: EMA(9,21) + Stochastic(14,3)
    #   Score=1.3977  Return=+67%  WR=44.8%  PF=1.55  Sharpe=3.25  MaxDD=-21%
    #   ETH correlation filter: negligible impact (kept for extra safety)
    strat = EMAScalpingStrategy(
        capital          = INITIAL_CAPITAL,
        risk_pct         = 0.015,        # 1.5% risk per trade
        ema_fast         = 9,
        ema_slow         = 21,           # wider gap vs 15 — fewer false crossovers
        ema_htf          = 50,
        atr_period       = 14,
        atr_sl_mult      = 1.5,          # wide SL — survives Binance tick noise
        slope_threshold  = 0.15,         # entry trend slope filter
        slope_lookback   = 5,            # 25 min window
        ema_prox_pct     = 0.003,
        volume_mult      = 1.2,          # entry candle must be above-average volume
        volume_period    = 20,
        max_rr_sl_atr    = 3.0,
        consec_loss_max  = 3,
        consec_cooldown  = 12,
        max_trades_day   = 5,
        daily_loss_pct   = 0.03,
        daily_profit_pct = 0.04,
        time_filter      = True,
        rr_ratio         = 2.0,
        use_stoch        = True,         # Stochastic(14,3) momentum gate
        stoch_k          = 14,
        stoch_d          = 3,
        use_ema_reclaim  = False,        # disabled — low WR across all timeframes
        use_engulfing    = False,        # disabled — 31.8% WR / -$97 over 365d
        blocked_hours    = {9,10,12,13,14,15},  # EU morning session — 10% WR, -$520 loss over 365d
    )

    print("Running strategy...")
    trades = strat.run(btc, eth)
    m = compute_metrics(trades, INITIAL_CAPITAL)
    print_report(m, INITIAL_CAPITAL)

    print("Generating charts...")
    plot_dashboard(m, INITIAL_CAPITAL, save_path="backtest_results.png")
    if not trades.empty:
        plot_price_chart(btc, trades, strat, last_n_days=7, save_path="price_chart.png")
