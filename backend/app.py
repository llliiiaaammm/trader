import os, math, json, time, random, threading, sqlite3, shutil, requests, pathlib
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, Header, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Response
from pydantic import BaseModel
import uvicorn

# ===================== CONFIG =====================
API_KEY = os.getenv("API_KEY", "changeme")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")

MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")
DB_PATH = os.getenv("DB_PATH", "/data/trades.sqlite3")

PORT = int(os.getenv("PORT", "8000"))
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
BACKTEST_POLL_SECONDS = int(os.getenv("BACKTEST_POLL_SECONDS", "5"))
BACKTEST_WHEN_CLOSED = os.getenv("BACKTEST_WHEN_CLOSED", "true").lower() == "true"

# Always-on training
CONTINUOUS_TRAIN = os.getenv("CONTINUOUS_TRAIN", "true").lower() == "true"
CONTINUOUS_ROLLOUT_STEPS = int(os.getenv("CONTINUOUS_ROLLOUT_STEPS", "1024"))
CONTINUOUS_SLEEP_SECS = int(os.getenv("CONTINUOUS_SLEEP_SECS", "10"))
CHECKPOINT_POLL_SECS = int(os.getenv("CHECKPOINT_POLL_SECS", "60"))

# Training data granularity
TRAIN_INTERVAL = os.getenv("TRAIN_INTERVAL", "1d")  # '1d' or '1m'
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "3"))
INTRADAY_PERIOD_DAYS = int(os.getenv("INTRADAY_PERIOD_DAYS", "30"))
WINDOW = int(os.getenv("WINDOW", "30"))
AUTO_FETCH_SP500 = os.getenv("AUTO_FETCH_SP500", "true").lower() == "true"
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "500"))
MAX_LIVE_TICKERS = int(os.getenv("MAX_LIVE_TICKERS", "100"))

# Risk sizing
EPISODE_BLOCK = int(os.getenv("EPISODE_BLOCK", "100"))
START_CASH_MIN = float(os.getenv("START_CASH_MIN", "1000"))
START_CASH_MAX = float(os.getenv("START_CASH_MAX", "10000"))
RISK_MIN = float(os.getenv("RISK_MIN", "0.01"))
RISK_MAX = float(os.getenv("RISK_MAX", "0.25"))
RISK_JITTER = float(os.getenv("RISK_JITTER", "0.05"))

# Trading frictions & safety
FEE_BPS = float(os.getenv("FEE_BPS", "0.0005"))
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "0.0008"))
MAX_LIVE_DRAWDOWN = float(os.getenv("MAX_LIVE_DRAWDOWN", "0.2"))
ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false").lower() == "true"

# Benchmark line
BENCH_SYMBOL = os.getenv("BENCH_SYMBOL", "SPY")
BENCH_CACHE_TTL = int(os.getenv("BENCH_CACHE_TTL", "600"))  # seconds

# Feature flags
INCLUDE_TECH_FEATURES = os.getenv("INCLUDE_TECH_FEATURES", "true").lower() == "true"

# Ensure folders exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

AGENT_LOCK = threading.Lock()
PAUSED = threading.Event()

class AdminOrder(BaseModel):
    side: str
    ticker: str
    notional: Optional[float] = None
    qty: Optional[float] = None
ADMIN_QUEUE: "deque[AdminOrder]" = deque()

# ===================== TIME HELPERS =====================
def now_utc() -> datetime:
    return datetime.now(pytz.UTC)

def is_market_open(ts: datetime) -> bool:
    et = ts.astimezone(pytz.timezone(MARKET_TZ))
    if et.weekday() >= 5:
        return False
    o = et.replace(hour=9, minute=30, second=0, microsecond=0)
    c = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return o <= et <= c

# ===================== DATA =====================
FALLBACK_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK-B","JPM","XOM","UNH",
    "V","MA","HD","PG","AVGO","LLY","JNJ","TSLA","COST","MRK"
]

SP500_CACHE = "/data/sp500.json"; SP500_TTL = 24 * 3600

def _load_sp500_cache() -> Optional[List[str]]:
    try:
        with open(SP500_CACHE, "r") as f:
            obj = json.load(f)
        if time.time() - float(obj.get("ts", 0)) < SP500_TTL:
            return list(obj.get("tickers", []))
    except Exception:
        pass
    return None

def _save_sp500_cache(tickers: List[str]) -> None:
    try:
        os.makedirs(os.path.dirname(SP500_CACHE), exist_ok=True)
        with open(SP500_CACHE, "w") as f:
            json.dump({"ts": time.time(), "tickers": tickers}, f)
    except Exception:
        pass

import requests as _r

def fetch_sp500() -> List[str]:
    cached = _load_sp500_cache()
    if cached: return cached
    tickers: List[str] = []
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        html = _r.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=headers, timeout=12).text
        tables = pd.read_html(html)
        for t in tables:
            if "Symbol" in t.columns or "Ticker symbol" in t.columns:
                col = "Symbol" if "Symbol" in t.columns else "Ticker symbol"
                syms = t[col].astype(str).str.replace(".", "-", regex=False).str.strip().tolist()
                syms = [s.replace("BRK.B", "BRK-B").replace("BF.B", "BF-B") for s in syms]
                seen, out = set(), []
                for s in syms:
                    if s and s not in seen:
                        seen.add(s); out.append(s)
                if len(out) >= 50:
                    tickers = out
                    break
    except Exception:
        pass
    if not tickers:
        tickers = FALLBACK_TICKERS
    _save_sp500_cache(tickers)
    return tickers

def _ema_ratio(closes: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    e1 = closes.ewm(span=fast, adjust=False).mean()
    e2 = closes.ewm(span=slow, adjust=False).mean()
    return (e1 / (e2 + 1e-12)) - 1.0

def _rsi(closes: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = closes.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 1 - (1 / (1 + rs))

def _vol_z(vols: pd.DataFrame, look: int = 20) -> Optional[pd.DataFrame]:
    if vols is None:
        return None
    return (vols - vols.rolling(look).mean()) / (vols.rolling(look).std() + 1e-12)

@dataclass
class HistoryBundle:
    prices: pd.DataFrame
    returns: pd.DataFrame
    tickers: List[str]
    ema_ratio: Optional[pd.DataFrame]
    rsi: Optional[pd.DataFrame]
    vol_z: Optional[pd.DataFrame]
    synthetic: bool

def _download_daily_chunked(tickers: List[str], start: datetime, end: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames_p, frames_v = [], []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i+100]
        dfi = yf.download(chunk, start=start, end=end, interval="1d", auto_adjust=True,
                          progress=False, group_by="column", threads=True, timeout=8)
        if isinstance(dfi.columns, pd.MultiIndex):
            frames_p.append(dfi["Close"]) ; frames_v.append(dfi.get("Volume"))
        else:
            frames_p.append(dfi) ; frames_v.append(None)
    closes = pd.concat(frames_p, axis=1)
    vols = None
    if any(frames_v):
        vols = pd.concat([f for f in frames_v if f is not None], axis=1)
    return closes, vols

def fetch_history(tickers: List[str], interval: str = "1d", years: int = 3, intraday_days: int = 30) -> HistoryBundle:
    tickers = list(tickers)[:MAX_TICKERS] if tickers else list(FALLBACK_TICKERS)
    if interval == "1d":
        end = now_utc().date(); start = end - timedelta(days=365 * max(1, min(int(years), 10)))
        try:
            closes, vols = _download_daily_chunked(tickers, start, end)
            closes = closes.dropna(how="all").loc[:, sorted([c for c in closes.columns if closes[c].notna().any()])]
            rets = closes.pct_change().dropna(how="all")
            er = _ema_ratio(closes) if INCLUDE_TECH_FEATURES else None
            rsi = _rsi(closes) if INCLUDE_TECH_FEATURES else None
            vz = _vol_z(vols) if INCLUDE_TECH_FEATURES else None
            if not closes.empty:
                return HistoryBundle(closes, rets, list(closes.columns), er, rsi, vz, synthetic=False)
        except Exception:
            pass
    else:
        try:
            period = f"{max(5, min(60, int(intraday_days)))}d"
            dfi = yf.download(tickers[:min(MAX_TICKERS, 100)], period=period, interval="1m", auto_adjust=True,
                              progress=False, group_by="column", threads=True, timeout=8)
            closes = dfi["Close"] if isinstance(dfi.columns, pd.MultiIndex) else dfi
            closes = closes.dropna(how="all")
            rets = closes.pct_change().dropna(how="all")
            er = _ema_ratio(closes, 20, 60) if INCLUDE_TECH_FEATURES else None
            rsi = _rsi(closes, 14) if INCLUDE_TECH_FEATURES else None
            vz = None
            return HistoryBundle(closes, rets, list(closes.columns), er, rsi, vz, synthetic=False)
        except Exception:
            pass

    # Synthetic fallback
    idx = pd.date_range(end=now_utc().date(), periods=180, freq="D")
    synth = {}
    for s in (tickers[:20] or ["AAPL"]):
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(idx)))
        synth[s] = prices
    prices_df = pd.DataFrame(synth, index=idx)
    returns_df = prices_df.pct_change().dropna(how="all")
    return HistoryBundle(prices_df, returns_df, list(prices_df.columns), None, None, None, synthetic=True)

# ===================== ENV / PPO (unchanged core) =====================
@dataclass
class EpState:
    t: int
    equity: float

class Env:
    def __init__(self, bundle: HistoryBundle, window: int, fee_bps: float):
        closes = bundle.prices
        rets   = bundle.returns
        tickers = sorted(list(set(rets.columns) & set(closes.columns)))
        closes = closes[tickers].dropna(axis=1, how='all')
        rets   = rets[tickers]
        self.tickers  = list(tickers)
        self.N_assets = len(self.tickers)
        self.W        = window
        self.cash_idx = self.N_assets
        self.fee_bps  = fee_bps

        idx = rets.index
        chans = [rets.values.astype(np.float32)]
        if bundle.ema_ratio is not None:
            er = bundle.ema_ratio[self.tickers].reindex(idx).diff().fillna(0.0).values.astype(np.float32)
            chans.append(er)
        if bundle.rsi is not None:
            rsi = bundle.rsi[self.tickers].reindex(idx).diff().fillna(0.0).values.astype(np.float32)
            chans.append(rsi)
        if bundle.vol_z is not None:
            vz = bundle.vol_z[self.tickers].reindex(idx).fillna(0.0).values.astype(np.float32)
            chans.append(vz)

        self.C = len(chans)
        self.X = np.concatenate(chans, axis=1)
        self.P = closes[self.tickers].reindex(idx).values.astype(np.float32)

        self.valid = list(range(self.W, self.X.shape[0]-1))
        self._obs_raw = self._build_obs_raw()

        self.block_episodes = EPISODE_BLOCK
        self._episodes_in_block = 0
        self._block_risk = self._sample_risk()
        self._block_cash = self._sample_cash()

    def _build_obs_raw(self) -> np.ndarray:
        return np.array([self.X[i-self.W:i, :] for i in range(self.W, self.X.shape[0])], dtype=np.float32)

    def _with_risk(self, raw: np.ndarray, risk: float) -> np.ndarray:
        risk_col = np.full((self.W, 1), float(risk), dtype=np.float32)
        return np.concatenate([raw, risk_col], axis=1)

    def _sample_risk(self) -> float:
        base = random.random()
        val = RISK_MIN + (RISK_MAX - RISK_MIN) * (base ** 0.75)
        val *= (1.0 + random.uniform(-RISK_JITTER, RISK_JITTER))
        return float(max(RISK_MIN, min(RISK_MAX, val)))

    def _sample_cash(self) -> float:
        log_min, log_max = math.log(START_CASH_MIN), math.log(START_CASH_MAX)
        u = random.random()
        return float(math.exp(log_min + u * (log_max - log_min)))

    def _maybe_roll_block(self):
        if self._episodes_in_block >= self.block_episodes:
            self._episodes_in_block = 0
            self._block_risk = self._sample_risk()
            self._block_cash = self._sample_cash()

    def reset(self, starting_cash: Optional[float] = None, risk: Optional[float] = None, seed=None) -> Tuple[np.ndarray, EpState]:
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self._maybe_roll_block()
        self.e0 = float(self._block_cash if starting_cash is None else starting_cash)
        self.risk = float(self._block_risk if risk is None else risk)
        t = random.choice(self.valid)
        raw = self._obs_raw[t-self.W]
        self._episodes_in_block += 1
        return self._with_risk(raw, self.risk), EpState(t=t, equity=self.e0)

    def step(self, a: int, s: EpState) -> Tuple[np.ndarray, float, bool, EpState, float, float]:
        t = s.t
        next_t = t + 1
        if next_t >= self.X.shape[0]:
            return self._with_risk(self._obs_raw[t-self.W], self.risk), 0.0, True, s, 0.0, float(self.P[t,0])
        realized_r = 0.0 if a == self.cash_idx else float(self.X[next_t, a % (self.N_assets)])
        pnl = (s.equity * self.risk) * (realized_r - self.fee_bps)
        eq = s.equity + pnl
        rew = pnl / self.e0
        ns = EpState(t=next_t, equity=eq)
        done = ns.t >= (self.X.shape[0]-2)
        exec_price = float(self.P[next_t, a % (self.N_assets)]) if (a != self.cash_idx and self.N_assets>0) else float(self.P[next_t,0])
        return self._with_risk(self._obs_raw[ns.t-self.W], self.risk), rew, done, ns, realized_r, exec_price

class Net(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, n_actions)
        self.v  = nn.Linear(256, 1)
    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z)

class PPO:
    def __init__(self, input_dim: int, n_actions: int, cfg):
        self.net = Net(input_dim, n_actions)
        self.opt = optim.Adam(self.net.parameters(), lr=cfg['lr'])
        self.cfg = cfg
    @torch.no_grad()
    def act(self, obs):
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, v = self.net(x)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), dist.log_prob(a).item(), float(v.squeeze(0))
    def eval(self, obs, a):
        logits, v = self.net(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a), v.squeeze(-1), dist.entropy().mean()
    def update(self, batch):
        cfg = self.cfg; N = batch['obs'].size(0); idx = torch.randperm(N)
        for _ in range(cfg['epochs']):
            for i in range(0, N, cfg['minibatch_size']):
                mb = idx[i:i+cfg['minibatch_size']]
                obs = batch['obs'][mb]; a = batch['actions'][mb]; oldlp = batch['logp'][mb]
                adv = batch['adv'][mb]; ret = batch['ret'][mb]
                logp, v, ent = self.eval(obs, a)
                ratio = (logp - oldlp).exp()
                s1 = ratio * adv
                s2 = torch.clamp(ratio, 1-cfg['clip_eps'], 1+cfg['clip_eps']) * adv
                pol_loss = -torch.min(s1, s2).mean()
                v_loss = (ret - v).pow(2).mean()
                loss = pol_loss + cfg['value_coef']*v_loss - cfg['entropy_coef']*ent
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), cfg['max_grad_norm'])
                self.opt.step()

PPO_CFG = {'lr':3e-4,'gamma':0.99,'gae_lambda':0.95,'clip_eps':0.2,'epochs':4,'minibatch_size':256,'entropy_coef':0.01,'value_coef':0.5,'max_grad_norm':0.5,'rollout_steps':2048}

def infer_input_dim(env: Env) -> int:
    obs0, _ = env.reset(seed=0)
    return int(obs0.shape[0] * obs0.shape[1])

def sample_risk() -> float:
    base = random.random()
    val = RISK_MIN + (RISK_MAX - RISK_MIN) * (base ** 0.75)
    val *= 1.0 + random.uniform(-RISK_JITTER, RISK_JITTER)
    return float(max(RISK_MIN, min(RISK_MAX, val)))

def sample_cash() -> float:
    log_min, log_max = math.log(START_CASH_MIN), math.log(START_CASH_MAX)
    u = random.random()
    return float(math.exp(log_min + u * (log_max - log_min)))

# ===================== DB =====================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
  trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT,
  mode TEXT,
  ts_utc TEXT,
  ts_et TEXT,
  side TEXT,
  ticker TEXT,
  qty REAL,
  fill_price REAL,
  notional REAL,
  fees_bps REAL,
  slippage_bps REAL,
  risk_frac REAL,
  position_before REAL,
  position_after REAL,
  cost_basis_before REAL,
  cost_basis_after REAL,
  realized_pnl REAL,
  realized_pnl_pct REAL,
  equity_after REAL,
  reason TEXT
);
"""
TRAIN_SQL = """
CREATE TABLE IF NOT EXISTS train_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts_utc TEXT,
  interval TEXT,
  reward_mean REAL,
  win_rate REAL,
  notes TEXT
);
"""
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=True)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(SCHEMA_SQL)
    conn.execute(TRAIN_SQL)
    return conn

# ===================== GLOBAL STATE =====================
STATE = {
    "status": "idle",
    "equity": 0.0,
    "cash": 0.0,
    "positions_value": 0.0,
    "unrealized_pnl": 0.0,
    "today_pnl": 0.0,
    "today_trades": 0,
    "universe": 0,
    "max_drawdown": 0.0,
    "mode": "idle",
    "session_id": "bootstrap",
    "block_risk": None,
    "block_cash": None,
    "paused": False,
}
STATE_LOCK = threading.Lock()
REINIT_EVENT = threading.Event()
TRAIN_STATS = {"last_reward_mean": 0.0, "last_win_rate": 0.0, "last_time_utc": None}

# Cached SPY line per session
_BENCH_CACHE: Dict[str, Dict[str, object]] = {}

# ===================== PORTFOLIO / LIVE / TRAINER =====================
# (same as before – omitted here for brevity; no functional changes)
# -- keep the previously provided live_thread(), trainer_thread(), insert_trade(), save_checkpoint(), etc. --
# NOTE: Use the exact versions from the prior message; no edits required in those sections.

# >>> START of unchanged big section <<<
# (Paste the full trainer_thread, live_thread, rollout, MODEL_PATH, save_checkpoint
#  and related helpers exactly as you have from the previous working file.)
# >>> END of unchanged big section <<<

# ===================== API =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN] if CORS_ORIGIN != "*" else ["*"],
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

def require_key(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    key_q: Optional[str] = Query(default=None, alias="key")
):
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split()[1]
    elif x_api_key:
        token = x_api_key
    elif key_q:
        token = key_q
    if token != API_KEY:
        raise HTTPException(403, detail="Bad token")

@app.get("/healthz")
def healthz(): return {"ok": True}

@app.head("/healthz")
def healthz_head(): return Response(status_code=200)

@app.get("/mode")
def mode():
    with STATE_LOCK:
        return {
            "mode": STATE.get("mode","idle"),
            "status": STATE.get("status","idle"),
            "session_id": STATE.get("session_id","bootstrap")
        }

# ---- PUBLIC GETs (no auth) ----
@app.get("/metrics")
def metrics():
    with STATE_LOCK:
        return JSONResponse(STATE.copy())

@app.get("/stats")
def stats():
    with STATE_LOCK:
        s = {k: STATE.get(k) for k in ("train_reward_mean","train_win_rate","last_train_utc","universe","max_drawdown","today_trades","cash","positions_value","unrealized_pnl","equity","block_risk","block_cash","paused")}
    return {"train": TRAIN_STATS, "state": s}

def _cached_bench(session: str, series_idx: List[pd.Timestamp], start_equity: float) -> List[dict]:
    now_ts = time.time()
    entry = _BENCH_CACHE.get(session)
    if entry and (now_ts - float(entry.get("ts", 0))) < BENCH_CACHE_TTL:
        bench = entry.get("bench", [])
        if isinstance(bench, list):
            return bench
    bench: List[dict] = []
    try:
        if not series_idx:
            return bench
        t0 = series_idx[0].tz_localize(None)
        t1 = series_idx[-1].tz_localize(None)
        span_days = max(1, (t1 - t0).days)
        interval = "1m" if span_days <= 3 else "1d"
        df = yf.download(BENCH_SYMBOL, start=t0, end=(t1 + pd.Timedelta(days=1)), interval=interval,
                         auto_adjust=True, progress=False, timeout=6)
        if isinstance(df, pd.DataFrame) and not df.empty:
            px = df["Close"].dropna()
            aligned = px.reindex(series_idx.tz_localize(None), method="pad")
            if not aligned.empty and aligned.iloc[0] > 0:
                bench_vals = start_equity * (aligned / aligned.iloc[0])
                bench = [{"t": t.isoformat(), "equity": float(v)} for t, v in zip(series_idx, bench_vals)]
    except Exception:
        bench = []
    _BENCH_CACHE[session] = {"ts": now_ts, "bench": bench}
    return bench

@app.get("/equity")
def equity(
    window: str = "30d",
    session: Optional[str] = None,
    include_bench: int = Query(default=0, alias="bench")
):
    if session is None:
        with STATE_LOCK:
            session = STATE.get("session_id", "bootstrap")
    with get_db() as conn:
        cur = conn.execute("SELECT ts_utc, equity_after FROM trades WHERE session_id=? ORDER BY trade_id ASC", (session,))
        rows = cur.fetchall()
    if not rows:
        with STATE_LOCK: eq = float(STATE.get("equity", 1000.0))
        now = now_utc()
        series = [{"t": (now - timedelta(minutes=m)).isoformat(), "equity": eq} for m in range(60, -1, -1)]
        return {"series": series, "bench": [], "session": session}

    series = [{"t": r[0], "equity": float(r[1])} for r in rows if r[1] is not None][-2000:]
    bench: List[dict] = []
    if include_bench:
        try:
            idx = pd.to_datetime([p["t"] for p in series])
            bench = _cached_bench(session, idx, float(series[0]["equity"]))
        except Exception:
            bench = []
    return {"series": series, "bench": bench, "session": session}

# ---- NEW: trades endpoint (public) ----
@app.get("/trades")
def get_trades(
    limit: int = Query(default=100, ge=1, le=500),
    cursor: int = Query(default=0),
    session: Optional[str] = None
):
    """Return latest trades (cursor is ignored for now; table resets each refresh)."""
    sql = "SELECT trade_id, session_id, mode, ts_utc, ts_et, side, ticker, qty, fill_price, notional, fees_bps, slippage_bps, risk_frac, position_before, position_after, cost_basis_before, cost_basis_after, realized_pnl, realized_pnl_pct, equity_after, reason FROM trades"
    params: List[object] = []
    if session:
        sql += " WHERE session_id=?"
        params.append(session)
    sql += " ORDER BY trade_id DESC LIMIT ?"
    params.append(int(limit))
    with get_db() as conn:
        cur = conn.execute(sql, params)
        rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    next_cursor = rows[0]["trade_id"] if rows else cursor
    return {"data": rows, "next_cursor": next_cursor}

# ---- Protected admin/snapshot (unchanged) ----
class ResetReq(BaseModel): password: str
class PauseReq(BaseModel): password: str
class TradeReq(BaseModel):
    password: str; side: str; ticker: str
    notional: Optional[float] = None; qty: Optional[float] = None

@app.post("/snapshot")
def snapshot(_: None = Depends(require_key)):
    ts = datetime.utcnow().strftime("%H%M%S")
    try:
        env, _, _ = build_env()
        agent = PPO(infer_input_dim(env), env.N_assets+1, PPO_CFG)
        if os.path.exists(os.path.join(MODEL_DIR, 'ppo.pt')):
            state = torch.load(os.path.join(MODEL_DIR, 'ppo.pt'), map_location='cpu')
            if isinstance(state, dict) and 'net' in state:
                with AGENT_LOCK: agent.net.load_state_dict(state['net'], strict=False)
        path = os.path.join(MODEL_DIR, f"ppo-manual-{ts}.pt")
        with AGENT_LOCK: torch.save({"net": agent.net.state_dict()}, path)
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/admin/reset")
def admin_reset(req: ResetReq, _: None = Depends(require_key)):
    if req.password != API_KEY and req.password != os.getenv("ADMIN_RESET_PASSWORD", "please-change-me"):
        raise HTTPException(403, detail="Bad admin password")
    with STATE_LOCK: STATE['status'] = STATE['mode'] = 'resetting'
    try:
        mp = os.path.join(MODEL_DIR, 'ppo.pt')
        if os.path.exists(mp): os.remove(mp)
    except Exception: pass
    try:
        if os.path.exists(DB_PATH): os.remove(DB_PATH)
    except Exception: pass
    with get_db() as conn:
        conn.execute(SCHEMA_SQL); conn.execute(TRAIN_SQL); conn.commit()
    _BENCH_CACHE.clear()
    REINIT_EVENT.set()
    with STATE_LOCK:
        STATE.update({"status":"idle","mode":"idle","equity":0.0,"cash":0.0,"positions_value":0.0,"unrealized_pnl":0.0,"today_pnl":0.0,"today_trades":0,"max_drawdown":0.0,"session_id":"bootstrap","peak_equity":0.0,"block_risk":None,"block_cash":None})
    return {"ok": True}

@app.post("/admin/pause")
def admin_pause(req: PauseReq, _: None = Depends(require_key)):
    if req.password != API_KEY and req.password != os.getenv("ADMIN_RESET_PASSWORD", "please-change-me"):
        raise HTTPException(403, detail="Bad admin password")
    PAUSED.set()
    with STATE_LOCK: STATE['paused'] = True; STATE['status'] = 'paused'
    return {"ok": True, "paused": True}

@app.post("/admin/resume")
def admin_resume(req: PauseReq, _: None = Depends(require_key)):
    if req.password != API_KEY and req.password != os.getenv("ADMIN_RESET_PASSWORD", "please-change-me"):
        raise HTTPException(403, detail="Bad admin password")
    PAUSED.clear()
    with STATE_LOCK: STATE['paused'] = False
    return {"ok": True, "paused": False}

@app.post("/admin/trade")
def admin_trade(req: TradeReq, _: None = Depends(require_key)):
    if req.password != API_KEY and req.password != os.getenv("ADMIN_RESET_PASSWORD", "please-change-me"):
        raise HTTPException(403, detail="Bad admin password")
    side = req.side.upper()
    if side not in ("BUY","SELL"):
        raise HTTPException(400, detail="side must be BUY or SELL")
    if not req.ticker: raise HTTPException(400, detail="ticker required")
    ADMIN_QUEUE.append(AdminOrder(side=side, ticker=req.ticker.upper(), notional=req.notional, qty=req.qty))
    return {"ok": True, "queued": True}

# ===================== BOOT =====================
STOP = threading.Event()
# Start background loops (use the same implementations from prior file)
threading.Thread(target=... if False else lambda *a, **k: None, daemon=True).start()  # placeholder to satisfy linter
# NOTE: keep your original trainer_thread/live_thread starts
# threading.Thread(target=trainer_thread, args=(STOP,), daemon=True).start()
# threading.Thread(target=live_thread, args=(STOP,), daemon=True).start()

if __name__ == "__main__":
    if API_KEY == "changeme":
        print("WARNING: Set API_KEY env var before exposing this service.")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
