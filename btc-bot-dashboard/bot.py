"""
Cloud Run Bot — BTC/ETH 5m Dual Strategy Forward Test
=====================================================
Stateless Flask handler triggered every 5 minutes by Cloud Scheduler.
State (capital, open trade, trade log) persisted in Firestore.

Runs BOTH strategies:
  1. Consensus ≥4/5 (original)
  2. V2 EMA/Stoch 5-Layer (trailing stop, partial TP, adaptive risk)

Architecture:
  Cloud Scheduler  →  POST /run  →  Cloud Run (this file)
                                          ↕
                                      Firestore
                                    (state + trades)

Deploy:
  gcloud run deploy btc-bot --source . --region us-central1 --no-allow-unauthenticated
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
#  Setup                                                           #
# ─────────────────────────────────────────────────────────────── #

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

app  = Flask(__name__)
db   = firestore.Client()

STATE_DOC  = db.collection('bot').document('state')
TRADES_COL = db.collection('trades')
RUNS_COL   = db.collection('runs')

INITIAL_CAPITAL = float(os.environ.get('INITIAL_CAPITAL', '1000'))

# ─────────────────────────────────────────────────────────────── #
#  Firestore state helpers                                         #
# ─────────────────────────────────────────────────────────────── #

def load_state() -> dict:
    doc = STATE_DOC.get()
    if doc.exists:
        return doc.to_dict()
    state = dict(
        capital=INITIAL_CAPITAL,
        active=None,
        trades_today=0,
        pnl_today=0.0,
        last_date=None,
        candles_held=0,
        trade_history=[],      # For adaptive risk
        equity_history=[INITIAL_CAPITAL],
    )
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
    """
    Called on every Cloud Scheduler trigger (every 5 min).
    Returns a dict describing what happened this tick.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    state = load_state()

    capital = state['capital']
    active  = state.get('active')
    today   = now.date().isoformat()

    # Reset daily counters on new day
    if state.get('last_date') != today:
        state['trades_today'] = 0
        state['pnl_today']    = 0.0
        state['last_date']    = today

    # ── Fetch live candles ────────────────────────────────────────
    try:
        btc_raw = fetch_candles('BTC/USDT')
        eth_raw = fetch_candles('ETH/USDT')
        shared  = set(btc_raw['datetime']) & set(eth_raw['datetime'])
        btc = btc_raw[btc_raw['datetime'].isin(shared)].reset_index(drop=True)
        eth = eth_raw[eth_raw['datetime'].isin(shared)].reset_index(drop=True)
    except Exception as e:
        log.error(f'Fetch failed: {e}')
        return {'status': 'error', 'message': str(e)}

    last       = btc.iloc[-1]
    btc_price  = float(last['close'])
    eth_price  = float(eth.iloc[-1]['close'])
    candle_ts  = last['datetime'].isoformat()

    # Compute ATR for exit management
    atr_series = v2_atr(btc, V2_CONFIG['atr_period'])
    atr_val = float(atr_series.iloc[-1])

    result = {
        'candle':    candle_ts,
        'btc_price': btc_price,
        'eth_price': eth_price,
        'atr':       round(atr_val, 2),
        'hour_utc':  now.hour,
        'action':    'none',
    }

    # ── Manage open trade ─────────────────────────────────────────
    if active is not None:
        candles_held = state.get('candles_held', 0) + 1
        state['candles_held'] = candles_held

        exit_px, pnl = check_exit(active, last, candles_held, atr_val)
        if pnl is not None:
            capital      += pnl
            status        = 'WIN' if pnl > 0 else 'LOSS'
            state['pnl_today']    += pnl
            state['trades_today'] += 1

            record = dict(
                entry_ts     = active['entry_ts'],
                exit_ts      = candle_ts,
                signal       = active.get('signal', 'BUY' if active['sig'] == BUY else 'SELL'),
                entry_price  = active['entry_px'],
                exit_price   = float(exit_px),
                sl           = active['sl'],
                initial_sl   = active.get('initial_sl', active['sl']),
                target       = active['tgt'],
                risk_usd     = active['risk_usd'],
                pnl          = float(pnl),
                status       = status,
                capital      = float(capital),
                strategy     = active.get('strategy', 'consensus'),
                confidence   = active.get('confidence', ''),
                candle_type  = active.get('candle_type', ''),
                breakeven_hit    = active.get('breakeven_hit', False),
                trailing_active  = active.get('trailing_active', False),
                remaining_pct    = active.get('remaining_pct', 1.0),
                votes        = active.get('votes', 0),
            )
            save_trade(record)
            log.info(f"Trade closed: {status}  pnl=${pnl:.2f}  capital=${capital:.2f}  "
                     f"strategy={active.get('strategy', '?')}")

            # Update adaptive risk history
            trade_history = state.get('trade_history', [])
            trade_history.append({'pnl': float(pnl), 'status': status})
            # Keep last 50 trades for adaptive risk
            state['trade_history'] = trade_history[-50:]

            equity_history = state.get('equity_history', [INITIAL_CAPITAL])
            equity_history.append(float(capital))
            state['equity_history'] = equity_history[-100:]

            state['active']  = None
            state['capital'] = capital
            state['candles_held'] = 0
            save_state(state)

            result['action']   = f'closed_{status.lower()}'
            result['pnl']      = round(pnl, 2)
            result['capital']  = round(capital, 2)
            result['strategy'] = active.get('strategy', '?')
            return result

        # Trade still open — save potentially updated SL (from trailing)
        state['active'] = active
        log.info(f"Trade open: {active.get('signal','?')}  entry={active['entry_px']:.2f}  "
                 f"sl={active['sl']:.2f}  tgt={active['tgt']:.2f}  btc={btc_price:.2f}  "
                 f"be={active.get('breakeven_hit',False)}  trail={active.get('trailing_active',False)}")
        result['action']   = 'holding'
        result['signal']   = active.get('signal', '?')
        result['entry_px'] = active['entry_px']
        result['breakeven_hit'] = active.get('breakeven_hit', False)
        result['trailing_active'] = active.get('trailing_active', False)
        save_state(state)
        return result

    # ── Daily limits ──────────────────────────────────────────────
    if state['trades_today'] >= V2_CONFIG['max_trades_day']:
        result['action'] = 'skip_max_trades'
        return result
    if state['pnl_today'] <= -(capital * V2_CONFIG['daily_loss_pct']):
        result['action'] = 'skip_daily_loss_limit'
        return result
    if state['pnl_today'] >= capital * V2_CONFIG['daily_profit_pct']:
        result['action'] = 'skip_daily_profit_limit'
        return result

    # ── Check for new signal (dual strategy) ──────────────────────
    direction, info = check_signal(btc, eth)
    if direction is not None:
        result['signal']            = 'BUY' if direction == 'long' else 'SELL'
        result['signal_strategy']   = info.get('strategy', '')
        result['signal_confidence'] = info.get('confidence', '')
        result['signal_votes']      = info.get('votes', 0)
    if direction is None:
        log.info(f"No signal  btc=${btc_price:.2f}")
        save_state(state)
        return result

    # Build trade using V2 builder (has advanced fields)
    df = btc.reset_index(drop=True)
    trade_history = state.get('trade_history', [])
    equity_history = state.get('equity_history', [INITIAL_CAPITAL])

    trade = v2_build_trade(
        BUY if direction == 'long' else SELL,
        df, atr_series, len(df) - 1, capital,
        trade_history, equity_history,
    )

    if trade is None:
        result['action'] = 'signal_invalid_trade'
        save_state(state)
        return result

    # Open new paper trade
    strategy = info.get('strategy', 'consensus')
    active = {
        **trade,
        'entry_ts'   : candle_ts,
        'strategy'   : strategy,
        'confidence' : info.get('confidence', ''),
        'votes'      : info.get('votes', 0),
        'candle_type': info.get('candle_type', ''),
        # Ensure Firestore-safe types
        'entry_px'   : float(trade['entry_px']),
        'sl'         : float(trade['sl']),
        'initial_sl' : float(trade['initial_sl']),
        'tgt'        : float(trade['tgt']),
        'risk_usd'   : float(trade['risk_usd']),
        'rpu'        : float(trade['rpu']),
        'sig'        : int(trade['sig']),
    }
    state['active']  = active
    state['capital'] = capital
    state['candles_held'] = 0
    save_state(state)

    log.info(f"Trade opened: {active.get('signal','?')}  entry={active['entry_px']:.2f}  "
             f"sl={active['sl']:.2f}  tgt={active['tgt']:.2f}  "
             f"strategy={strategy}  confidence={info.get('confidence', '?')}")

    result['action']     = f"opened_{direction}"
    result['signal']     = active.get('signal', '?')
    result['entry_px']   = active['entry_px']
    result['sl']         = active['sl']
    result['tgt']        = active['tgt']
    result['strategy']   = strategy
    result['confidence'] = info.get('confidence', '')
    result['votes']      = info.get('votes', 0)
    result['candle_type']= info.get('candle_type', '')
    return result


# ─────────────────────────────────────────────────────────────── #
#  Flask routes                                                    #
# ─────────────────────────────────────────────────────────────── #

@app.route('/run', methods=['POST'])
def run_tick():
    """Called by Cloud Scheduler every 5 minutes."""
    try:
        result = tick()
        log.info(f"Tick result: {result}")
        try:
            if isinstance(result, dict) and result.get('status') != 'error':
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
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp


@app.route('/status', methods=['GET', 'OPTIONS'])
def status():
    """Health check + current state — public read-only."""
    if request.method == 'OPTIONS':
        return _cors({})
    try:
        state = load_state()
        active = state.get('active')

        return _cors({
            'status'      : 'ok',
            'capital'     : round(state['capital'], 2),
            'initial'     : INITIAL_CAPITAL,
            'return_pct'  : round((state['capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
            'active'      : active,
            'trades_today': state.get('trades_today', 0),
            'pnl_today'   : round(state.get('pnl_today', 0), 2),
            'last_date'   : state.get('last_date'),
            'strategy_mode': 'dual',
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
        docs   = TRADES_COL.order_by('entry_ts', direction=firestore.Query.DESCENDING).limit(50).stream()
        trades = [d.to_dict() for d in docs]
        wins   = sum(1 for t in trades if t.get('pnl', 0) > 0)
        total_pnl = sum(t.get('pnl', 0) for t in trades)

        # Strategy breakdown
        by_strategy = {}
        for t in trades:
            s = t.get('strategy', 'consensus')
            if s not in by_strategy:
                by_strategy[s] = {'count': 0, 'wins': 0, 'pnl': 0}
            by_strategy[s]['count'] += 1
            if t.get('pnl', 0) > 0:
                by_strategy[s]['wins'] += 1
            by_strategy[s]['pnl'] += t.get('pnl', 0)

        return _cors({
            'count'    : len(trades),
            'wins'     : wins,
            'losses'   : len(trades) - wins,
            'win_rate' : round(wins / len(trades) * 100, 1) if trades else 0,
            'total_pnl': round(total_pnl, 2),
            'trades'   : trades,
            'strategies': by_strategy,
        })
    except Exception as e:
        return _cors({'status': 'error', 'message': str(e)}, 500)


# ─────────────────────────────────────────────────────────────── #
#  Entry point                                                     #
# ─────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
