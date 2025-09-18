import os, math, json, time, random, threading, sqlite3
# In this paper demo we treat every minute as a full open-close for PnL on a notional slice
notional = equity*risk
qty = notional / max(price,1e-6)
fee = notional*FEE_BPS
pnl = notional*(r) - fee
equity += pnl
side = 'BUY' if r>=0 else 'SELL' # simplistic labeling for demo
# Track realized pnl on SELL-like actions
realized = pnl if side=='SELL' else 0.0
realized_pct = realized/notional if notional>0 else 0.0
et = now_utc().astimezone(pytz.timezone(MARKET_TZ))
log_trade(
session_id="live", mode="LIVE", ts_utc=now_utc().isoformat(), ts_et=et.isoformat(),
side=side, ticker=act_ticker, qty=qty, fill_price=price, notional=notional, fees_bps=FEE_BPS,
slippage_bps=0.0, risk_frac=risk, position_before=0.0, position_after=qty,
cost_basis_before=0.0, cost_basis_after=price, realized_pnl=realized,
realized_pnl_pct=realized_pct, equity_after=equity, reason="policy argmax"
)
with STATE_LOCK:
STATE['today_trades'] += 1
today_et_now = now_utc().astimezone(pytz.timezone(MARKET_TZ)).date()
if today_et_now != today_et: today_et=today_et_now; today_pnl=0.0; with STATE_LOCK: STATE['today_trades']=0
today_pnl += pnl
with STATE_LOCK:
STATE['equity']=float(equity); STATE['today_pnl']=float(today_pnl)


time.sleep(POLL_SECONDS)
except Exception as e:
with STATE_LOCK: STATE['status']=f"error: {e}"; time.sleep(POLL_SECONDS)


# ------------------ API ------------------
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=[CORS_ORIGIN] if CORS_ORIGIN!="*" else ["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


def require_key(authorization: Optional[str] = Header(default=None)):
if not authorization or not authorization.lower().startswith("bearer "):
raise HTTPException(401, detail="Missing bearer token")
token = authorization.split()[1]
if token != API_KEY: raise HTTPException(403, detail="Bad token")


@app.get("/healthz")
def healthz(): return {"ok": True}


@app.get("/metrics")
def metrics(_: None = require_key()):
with STATE_LOCK: return JSONResponse(STATE.copy())


@app.get("/trades")
def trades(limit: int = 100, cursor: int = 0, _: None = require_key()):
limit = max(1, min(500, limit));
cur = DB.execute("SELECT * FROM trades WHERE trade_id>? ORDER BY trade_id DESC LIMIT ?", (cursor, limit))
rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
next_cursor = rows[-1]['trade_id'] if rows else cursor
return {"data": rows, "next_cursor": next_cursor}


@app.get("/equity")
def equity(window: str = "30d", _: None = require_key()):
# derive a simple synthetic equity curve from trades table (paper demo); in production, store timeseries
cur = DB.execute("SELECT ts_utc, equity_after FROM trades ORDER BY trade_id ASC")
rows = cur.fetchall()
series = [{"t": r[0], "equity": r[1]} for r in rows]
return {"series": series[-2000:]} # cap for payload size


# ------------------ BOOT ------------------
STOP = threading.Event()
threading.Thread(target=trainer_thread, args=(STOP,), daemon=True).start()
threading.Thread(target=live_thread, args=(STOP,), daemon=True).start()


if __name__ == "__main__":
uvicorn.run(app, host="0.0.0.0", port=PORT)
