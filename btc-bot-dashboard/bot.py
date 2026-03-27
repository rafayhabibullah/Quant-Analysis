"""
Cloud Run Bot — Multi-Asset Dual Strategy
==========================================
Stateless Flask handler triggered every 5 minutes by Cloud Scheduler.
State (capital, per-symbol trades) persisted in one Firestore doc.

Symbols (9 crypto pairs):
  BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT,
  SUI/USDT, OP/USDT, LINK/USDT, AVAX/USDT

Strategy per symbol:
  BTC/USDT → Dual: Consensus ≥4/5  +  V2 EMA/Stoch 5-Layer
  Others   → V2 EMA/Stoch 5-Layer only

Params (from trading-system 90-day optimization):
  EMA 9/21/50 · ATR_SL×0.75 · slope≥0.25 · vol≥1.3× · RR 2:1 · risk 2.5%

Architecture:
  Cloud Scheduler → POST /run → Cloud Run (this file)
                                      ↕
                                  Firestore
                          (bot/state — single doc holds all)

Deploy:
  gcloud run deploy btc-bot --source . --region europe-west1 --allow-unauthenticated
"""

import warnings
warnings.filterwarnings("ignore")

import os
import logging
from datetime import datetime, timezone
from flask import Flask, jsonify, make_response, request
from google.cloud import firestore

from forward_test import (
    fetch_candles, check_signal, check_exit, LOG_FILE,
)
from strategy_v2 import (
    CONFIG as V2_CONFIG, BUY, SELL,
    atr as v2_atr,
    build_trade as v2_build_trade,
)

# ─────────────────────────────────────────────────────────────── #
#  Config                                                          #
# ─────────────────────────────────────────────────────────────── #

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app = Flask(__name__)
db  = firestore.Client()

STATE_DOC  = db.collection('bot').document('state')
TRADES_COL = db.collection('trades')
RUNS_COL   = db.collection('runs')

INITIAL_CAPITAL    = float(os.environ.get('INITIAL_CAPITAL', '1000'))
MAX_OPEN_POSITIONS = 5

# Primary symbol → correlation pair
# Params validated via 90-day optimization in trading-system
SYMBOLS = {
    'BTC/USDT':  'ETH/USDT',
    'ETH/USDT':  'BTC/USDT',
    'SOL/USDT':  'ETH/USDT',
    'BNB/USDT':  'BTC/USDT',
    'XRP/USDT':  'BTC/USDT',
    'SUI/USDT':  'SOL/USDT',
    'OP/USDT':   'ETH/USDT',
    'LINK/USDT': 'ETH/USDT',
    'AVAX/USDT': 'ETH/USDT',
}


def _key(symbol: str) -> str:
    """Firestore-safe key: BTC/USDT → BTC_USDT"""
    return symbol.replace('/', '_')


def _empty_sym_state() -> dict:
    return {
        'active':       None,
        'trades_today': 0,
        'pnl_today':    0.0,
        'candles_held': 0,
        'last_date':    None,
    }


# ─────────────────────────────────────────────────────────────── #
#  Firestore helpers                                               #
# ─────────────────────────────────────────────────────────────── #

def load_state() -> dict:
    doc = STATE_DOC.get()
    if doc.exists:
        state = doc.to_dict()
        # Migrate: ensure new symbols are present
        syms = state.setdefault('symbols', {})
        for sym in SYMBOLS:
            syms.setdefault(_key(sym), _empty_sym_state())
        return state
    state = {
        'capital':        INITIAL_CAPITAL,
        'equity_history': [INITIAL_CAPITAL],
        'trade_history':  [],
        'symbols': {_key(sym): _empty_sym_state() for sym in SYMBOLS},
    }
    STATE_DOC.set(state)
    return state


def save_state(state: dict):
    STATE_DOC.set(state)


def save_trade(record: dict):
    TRADES_COL.add(record)


def save_run(record: dict):
    RUNS_COL.add(record)


# ─────────────────────────────────────────────────────────────── #
#  Core tick handler                                               #
# ─────────────────────────────────────────────────────────────── #

def tick() -> dict:
    now   = datetime.now(timezone.utc).replace(tzinfo=None)
    today = now.date().isoformat()
    state = load_state()
    capital = state['capital']

    # Fetch all unique symbols (primary + correlation, deduplicated)
    needed  = set(SYMBOLS.keys()) | set(SYMBOLS.values())
    candles = {}
    for sym in needed:
        try:
            candles[sym] = fetch_candles(sym)
        except Exception as e:
            log.warning(f'Fetch failed {sym}: {e}')

    # Count currently open positions
    open_count = sum(
        1 for sym in SYMBOLS
        if state['symbols'].get(_key(sym), {}).get('active') is not None
    )

    run_symbols = {}

    for sym, corr_sym in SYMBOLS.items():
        if sym not in candles or corr_sym not in candles:
            run_symbols[sym] = {'action': 'fetch_error'}
            continue

        # Align to shared timestamps
        prim_raw = candles[sym]
        corr_raw = candles[corr_sym]
        shared   = set(prim_raw['datetime']) & set(corr_raw['datetime'])
        prim     = prim_raw[prim_raw['datetime'].isin(shared)].reset_index(drop=True)
        corr     = corr_raw[corr_raw['datetime'].isin(shared)].reset_index(drop=True)

        last      = prim.iloc[-1]
        price     = float(last['close'])
        candle_ts = last['datetime'].isoformat()

        atr_series = v2_atr(prim, V2_CONFIG['atr_period'])
        atr_val    = float(atr_series.iloc[-1])

        sk = _key(sym)
        ss = state['symbols'][sk]

        # Reset daily counters on new day
        if ss.get('last_date') != today:
            ss['trades_today'] = 0
            ss['pnl_today']    = 0.0
            ss['last_date']    = today

        sym_result = {'price': price, 'atr': round(atr_val, 2), 'action': 'none'}

        # ── Manage open trade ──────────────────────────────────
        if ss.get('active'):
            candles_held       = ss.get('candles_held', 0) + 1
            ss['candles_held'] = candles_held
            active             = ss['active']

            exit_px, pnl = check_exit(active, last, candles_held, atr_val)

            if pnl is not None:
                capital          += pnl
                status            = 'WIN' if pnl > 0 else 'LOSS'
                ss['pnl_today']  += pnl
                ss['trades_today'] += 1

                record = dict(
                    symbol           = sym,
                    entry_ts         = active['entry_ts'],
                    exit_ts          = candle_ts,
                    signal           = active.get('signal', 'BUY' if active['sig'] == BUY else 'SELL'),
                    entry_price      = active['entry_px'],
                    exit_price       = float(exit_px),
                    sl               = active['sl'],
                    initial_sl       = active.get('initial_sl', active['sl']),
                    target           = active['tgt'],
                    risk_usd         = active['risk_usd'],
                    pnl              = float(pnl),
                    status           = status,
                    capital          = float(capital),
                    strategy         = active.get('strategy', 'v2'),
                    confidence       = active.get('confidence', ''),
                    candle_type      = active.get('candle_type', ''),
                    breakeven_hit    = active.get('breakeven_hit', False),
                    trailing_active  = active.get('trailing_active', False),
                    remaining_pct    = active.get('remaining_pct', 1.0),
                    votes            = active.get('votes', 0),
                )
                save_trade(record)
                log.info(f'{sym} CLOSED {status}  pnl=${pnl:.2f}  capital=${capital:.2f}')

                th = state.get('trade_history', [])
                th.append({'pnl': float(pnl), 'status': status})
                state['trade_history'] = th[-50:]

                eh = state.get('equity_history', [INITIAL_CAPITAL])
                eh.append(float(capital))
                state['equity_history'] = eh[-100:]

                ss['active']       = None
                ss['candles_held'] = 0
                open_count        -= 1

                sym_result['action'] = f'closed_{status.lower()}'
                sym_result['pnl']    = round(pnl, 2)
            else:
                ss['active'] = active   # Persist updated SL (trailing)
                sig_str = active.get('signal', 'BUY' if active['sig'] == BUY else 'SELL')
                sym_result['action']   = 'holding'
                sym_result['signal']   = sig_str
                sym_result['entry_px'] = active['entry_px']
                log.info(f'{sym} holding  entry={active["entry_px"]:.4f}  sl={active["sl"]:.4f}  price={price:.4f}')

            run_symbols[sym] = sym_result
            continue

        # ── Daily / position limits ────────────────────────────
        if ss['trades_today'] >= V2_CONFIG['max_trades_day']:
            sym_result['action'] = 'skip_max_trades'
            run_symbols[sym] = sym_result
            continue
        if ss['pnl_today'] <= -(capital * V2_CONFIG['daily_loss_pct']):
            sym_result['action'] = 'skip_daily_loss'
            run_symbols[sym] = sym_result
            continue
        if ss['pnl_today'] >= capital * V2_CONFIG['daily_profit_pct']:
            sym_result['action'] = 'skip_daily_profit'
            run_symbols[sym] = sym_result
            continue
        if open_count >= MAX_OPEN_POSITIONS:
            sym_result['action'] = 'skip_max_positions'
            run_symbols[sym] = sym_result
            continue

        # ── Signal check ───────────────────────────────────────
        # All pairs: dual strategy (Consensus ≥4/5 + V2)
        direction, info = check_signal(prim, corr)

        if direction is not None:
            sym_result['signal']            = 'BUY' if direction == 'long' else 'SELL'
            sym_result['signal_strategy']   = info.get('strategy', 'v2')
            sym_result['signal_confidence'] = info.get('confidence', '')
            sym_result['signal_votes']      = info.get('votes', 0)

        if direction is None:
            log.info(f'{sym} no signal  price={price:.4f}')
            run_symbols[sym] = sym_result
            continue

        # ── Build trade ────────────────────────────────────────
        df    = prim.reset_index(drop=True)
        trade = v2_build_trade(
            BUY if direction == 'long' else SELL,
            df, atr_series, len(df) - 1, capital,
            state.get('trade_history', []),
            state.get('equity_history', [INITIAL_CAPITAL]),
        )

        if trade is None:
            sym_result['action'] = 'signal_invalid_trade'
            run_symbols[sym] = sym_result
            continue

        strategy = info.get('strategy', 'v2')
        active = {
            **trade,
            'entry_ts'   : candle_ts,
            'signal'     : 'BUY' if direction == 'long' else 'SELL',
            'strategy'   : strategy,
            'confidence' : info.get('confidence', ''),
            'votes'      : info.get('votes', 0),
            'candle_type': info.get('candle_type', ''),
            # Firestore-safe types
            'entry_px'   : float(trade['entry_px']),
            'sl'         : float(trade['sl']),
            'initial_sl' : float(trade['initial_sl']),
            'tgt'        : float(trade['tgt']),
            'risk_usd'   : float(trade['risk_usd']),
            'rpu'        : float(trade['rpu']),
            'sig'        : int(trade['sig']),
        }
        ss['active']       = active
        ss['candles_held'] = 0
        open_count        += 1

        log.info(f'{sym} OPENED {strategy} {direction.upper()}  '
                 f'entry={active["entry_px"]:.4f}  sl={active["sl"]:.4f}  tgt={active["tgt"]:.4f}')

        sym_result['action']     = f'opened_{direction}'
        sym_result['entry_px']   = active['entry_px']
        sym_result['sl']         = active['sl']
        sym_result['tgt']        = active['tgt']
        sym_result['strategy']   = strategy
        sym_result['confidence'] = info.get('confidence', '')
        run_symbols[sym] = sym_result

    state['capital'] = capital
    save_state(state)

    # Use BTC candle timestamp as the run timestamp
    btc_df    = candles.get('BTC/USDT')
    candle_ts = btc_df.iloc[-1]['datetime'].isoformat() if btc_df is not None and len(btc_df) > 0 else now.isoformat()

    return {
        'candle'        : candle_ts,
        'capital'       : round(capital, 2),
        'open_positions': open_count,
        'hour_utc'      : now.hour,
        'symbols'       : run_symbols,
    }


# ─────────────────────────────────────────────────────────────── #
#  Flask routes                                                    #
# ─────────────────────────────────────────────────────────────── #

@app.route('/run', methods=['POST'])
def run_tick():
    """Called by Cloud Scheduler every 5 minutes."""
    try:
        result = tick()
        signals = [s for s, d in result.get('symbols', {}).items() if d.get('signal')]
        log.info(f"Tick: capital=${result.get('capital')}  "
                 f"open={result.get('open_positions')}  signals={signals or 'none'}")
        try:
            save_run(result)
        except Exception as e:
            log.warning(f'save_run error: {e}')
        return jsonify(result), 200
    except Exception as e:
        log.exception('Unhandled error in tick')
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _cors(data, code=200):
    """Wrap jsonify response with CORS headers for GitHub Pages."""
    resp = make_response(jsonify(data), code)
    resp.headers['Access-Control-Allow-Origin']  = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


@app.route('/status', methods=['GET', 'OPTIONS'])
def status():
    """Current state across all symbols — public read-only."""
    if request.method == 'OPTIONS':
        return _cors({})
    try:
        state = load_state()

        actives = {
            sym: state['symbols'][_key(sym)]['active']
            for sym in SYMBOLS
            if state['symbols'].get(_key(sym), {}).get('active') is not None
        }
        trades_today = sum(state['symbols'].get(_key(s), {}).get('trades_today', 0) for s in SYMBOLS)
        pnl_today    = sum(state['symbols'].get(_key(s), {}).get('pnl_today', 0.0) for s in SYMBOLS)

        return _cors({
            'status'        : 'ok',
            'capital'       : round(state['capital'], 2),
            'initial'       : INITIAL_CAPITAL,
            'return_pct'    : round((state['capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
            'active'        : actives,
            'open_positions': len(actives),
            'trades_today'  : trades_today,
            'pnl_today'     : round(pnl_today, 2),
            'last_date'     : state['symbols'].get('BTC_USDT', {}).get('last_date'),
            'strategy_mode' : 'dual_multi',
            'symbols'       : list(SYMBOLS.keys()),
        })
    except Exception as e:
        return _cors({'status': 'error', 'message': str(e)}, 500)


@app.route('/latest-run', methods=['GET', 'OPTIONS'])
def latest_run():
    """Most recent tick result — public read-only."""
    if request.method == 'OPTIONS':
        return _cors({})
    try:
        docs = RUNS_COL.order_by('candle', direction=firestore.Query.DESCENDING).limit(1).stream()
        runs = [d.to_dict() for d in docs]
        return _cors(runs[0] if runs else {})
    except Exception as e:
        return _cors({'error': str(e)}, 500)


@app.route('/runs', methods=['GET', 'OPTIONS'])
def get_runs():
    """Last 50 tick run logs — public read-only."""
    if request.method == 'OPTIONS':
        return _cors({})
    try:
        docs = RUNS_COL.order_by('candle', direction=firestore.Query.DESCENDING).limit(50).stream()
        runs = [d.to_dict() for d in docs]
        return _cors({'count': len(runs), 'runs': runs})
    except Exception as e:
        return _cors({'error': str(e)}, 500)


@app.route('/trades', methods=['GET', 'OPTIONS'])
def get_trades():
    """Return all logged trades from Firestore — public read-only."""
    if request.method == 'OPTIONS':
        return _cors({})
    try:
        docs   = TRADES_COL.order_by('entry_ts', direction=firestore.Query.DESCENDING).limit(100).stream()
        trades = [d.to_dict() for d in docs]
        wins      = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total_pnl = sum(t.get('pnl', 0) for t in trades)

        # Strategy breakdown
        by_strategy = {}
        for t in trades:
            s = t.get('strategy', 'v2')
            if s not in by_strategy:
                by_strategy[s] = {'count': 0, 'wins': 0, 'pnl': 0}
            by_strategy[s]['count'] += 1
            if t.get('pnl', 0) > 0:
                by_strategy[s]['wins'] += 1
            by_strategy[s]['pnl'] += t.get('pnl', 0)

        # Symbol breakdown
        by_symbol = {}
        for t in trades:
            sym = t.get('symbol', 'BTC/USDT')
            if sym not in by_symbol:
                by_symbol[sym] = {'count': 0, 'wins': 0, 'pnl': 0}
            by_symbol[sym]['count'] += 1
            if t.get('pnl', 0) > 0:
                by_symbol[sym]['wins'] += 1
            by_symbol[sym]['pnl'] += t.get('pnl', 0)

        return _cors({
            'count'     : len(trades),
            'wins'      : wins,
            'losses'    : len(trades) - wins,
            'win_rate'  : round(wins / len(trades) * 100, 1) if trades else 0,
            'total_pnl' : round(total_pnl, 2),
            'trades'    : trades,
            'strategies': by_strategy,
            'symbols'   : by_symbol,
        })
    except Exception as e:
        return _cors({'status': 'error', 'message': str(e)}, 500)


# ─────────────────────────────────────────────────────────────── #
#  Entry point                                                     #
# ─────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
