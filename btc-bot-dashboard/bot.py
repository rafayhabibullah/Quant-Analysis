"""
Cloud Run Bot — BTC/ETH 5m Forward Test
========================================
Stateless Flask handler triggered every 5 minutes by Cloud Scheduler.
State (capital, open trade, trade log) persisted in Firestore.

Architecture:
  Cloud Scheduler  →  POST /run  →  Cloud Run (this file)
                                          ↕
                                      Firestore
                                    (state + trades)

Deploy:
  gcloud run deploy btc-bot --source . --region us-central1 --no-allow-unauthenticated

Scheduler (run after deploy):
  gcloud scheduler jobs create http btc-bot-trigger \\
    --schedule="*/5 * * * *" \\
    --uri="https://<YOUR_CLOUD_RUN_URL>/run" \\
    --http-method=POST \\
    --oidc-service-account-email=<YOUR_SA>@<PROJECT>.iam.gserviceaccount.com \\
    --location=us-central1
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
from optimize_directional import FIXED, BUY, SELL, _atr
from backtest_multi import build_trade

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

INITIAL_CAPITAL = float(os.environ.get('INITIAL_CAPITAL', '1000'))

# ─────────────────────────────────────────────────────────────── #
#  Firestore state helpers                                         #
# ─────────────────────────────────────────────────────────────── #

def load_state() -> dict:
    doc = STATE_DOC.get()
    if doc.exists:
        return doc.to_dict()
    # First run — initialise
    state = dict(capital=INITIAL_CAPITAL, active=None,
                 trades_today=0, pnl_today=0.0, last_date=None)
    STATE_DOC.set(state)
    return state


def save_state(state: dict):
    STATE_DOC.set(state)


def save_trade(record: dict):
    TRADES_COL.add(record)

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
    active  = state.get('active')      # None or trade dict
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
    candle_ts  = last['datetime'].isoformat()

    result = {'candle': candle_ts, 'btc_price': btc_price, 'action': 'none'}

    # ── Manage open trade ─────────────────────────────────────────
    if active is not None:
        import pandas as pd
        exit_px, pnl = check_exit(active, last)
        if pnl is not None:
            capital      += pnl
            status        = 'WIN' if pnl > 0 else 'LOSS'
            state['pnl_today']    += pnl
            state['trades_today'] += 1

            record = dict(
                entry_ts    = active['entry_ts'],
                exit_ts     = candle_ts,
                signal      = 'BUY' if active['sig'] == BUY else 'SELL',
                entry_price = active['entry_px'],
                exit_price  = float(exit_px),
                sl          = active['sl'],
                target      = active['tgt'],
                risk_usd    = active['risk_usd'],
                pnl         = float(pnl),
                status      = status,
                capital     = float(capital),
                votes       = active.get('votes', 4),
            )
            save_trade(record)
            log.info(f"Trade closed: {status}  pnl=${pnl:.2f}  capital=${capital:.2f}")

            state['active']  = None
            state['capital'] = capital
            save_state(state)

            result['action'] = f'closed_{status.lower()}'
            result['pnl']    = round(pnl, 2)
            result['capital']= round(capital, 2)
            return result

        # Trade still open
        log.info(f"Trade open: {active['signal']}  entry={active['entry_px']:.2f}  "
                 f"sl={active['sl']:.2f}  tgt={active['tgt']:.2f}  btc={btc_price:.2f}")
        result['action']  = 'holding'
        result['signal']  = active.get('signal', '?')
        result['entry_px']= active['entry_px']
        save_state(state)
        return result

    # ── Daily limits ──────────────────────────────────────────────
    if state['trades_today'] >= FIXED['max_trades_day']:
        result['action'] = 'skip_max_trades'
        return result
    if state['pnl_today'] <= -(capital * FIXED['daily_loss_pct']):
        result['action'] = 'skip_daily_loss_limit'
        return result
    if state['pnl_today'] >= capital * FIXED['daily_profit_pct']:
        result['action'] = 'skip_daily_profit_limit'
        return result

    # ── Check for new signal ──────────────────────────────────────
    direction, combo = check_signal(btc, eth)
    if direction is None:
        log.info(f"No signal  btc=${btc_price:.2f}")
        save_state(state)
        return result

    df    = btc.reset_index(drop=True)
    atr   = _atr(df, FIXED['atr_period'])
    trade = build_trade(combo, df, atr, len(df)-1, capital, direction)

    if trade is None:
        result['action'] = 'signal_invalid_trade'
        save_state(state)
        return result

    # Open new paper trade
    active = {
        **trade,
        'signal'  : 'BUY' if trade['sig'] == BUY else 'SELL',
        'entry_ts': candle_ts,
        'votes'   : combo.get('votes', 4),
        # Firestore can't store numpy types — convert
        'entry_px': float(trade['entry_px']),
        'sl'      : float(trade['sl']),
        'tgt'     : float(trade['tgt']),
        'risk_usd': float(trade['risk_usd']),
        'rpu'     : float(trade['rpu']),
        'sig'     : int(trade['sig']),
    }
    state['active']  = active
    state['capital'] = capital
    save_state(state)

    log.info(f"Trade opened: {active['signal']}  entry={active['entry_px']:.2f}  "
             f"sl={active['sl']:.2f}  tgt={active['tgt']:.2f}  "
             f"votes={active['votes']}")

    result['action']   = f"opened_{direction}"
    result['signal']   = active['signal']
    result['entry_px'] = active['entry_px']
    result['sl']       = active['sl']
    result['tgt']      = active['tgt']
    result['votes']    = active['votes']
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
        return _cors({
            'status'      : 'ok',
            'capital'     : round(state['capital'], 2),
            'initial'     : INITIAL_CAPITAL,
            'return_pct'  : round((state['capital'] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 2),
            'active'      : state.get('active'),
            'trades_today': state.get('trades_today', 0),
            'pnl_today'   : round(state.get('pnl_today', 0), 2),
            'last_date'   : state.get('last_date'),
        })
    except Exception as e:
        return _cors({'status': 'error', 'message': str(e)}, 500)


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
        return _cors({
            'count'    : len(trades),
            'wins'     : wins,
            'losses'   : len(trades) - wins,
            'win_rate' : round(wins / len(trades) * 100, 1) if trades else 0,
            'total_pnl': round(total_pnl, 2),
            'trades'   : trades,
        })
    except Exception as e:
        return _cors({'status': 'error', 'message': str(e)}, 500)


# ─────────────────────────────────────────────────────────────── #
#  Entry point                                                     #
# ─────────────────────────────────────────────────────────────── #

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
