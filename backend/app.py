import os, math, json, time, random, threading, sqlite3, shutil, requests, pathlib
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
from fastapi import FastAPI, Header, HTTPException, Depends, Query
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
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "3"))            # used for 1d
INTRADAY_PERIOD_DAYS = int(os.getenv("INTRADAY_PERIOD_DAYS", "30"))  # used for 1m
WINDOW = int(os.getenv("WINDOW", "30"))   # lookback bars
AUTO_FETCH_SP500 = os.getenv("AUTO_FETCH_SP500", "true").lower() == "true"
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "500"))               # backtest/training
MAX_LIVE_TICKERS = int(os.getenv("MAX_LIVE_TICKERS", "100"))     # live minute polling cap

# Risk sizing
EPISODE_BLOCK = int(os.getenv("EPISODE_BLOCK", "100"))
START_CASH_MIN = float(os.getenv("START_CASH_MIN", "1000"))
START_CASH_MAX = float(os.getenv("START_CASH_MAX", "10000"))
RISK_MIN = float(os.getenv("RISK_MIN", "0.01"))
RISK_MAX = float(os.getenv("RISK_MAX", "0.25"))
RISK_JITTER = float(os.getenv("RISK_JITTER", "0.05"))

# Trading frictions & safety
FEE_BPS = float(os.getenv("FEE_BPS", "0.0005"))     # 5 bps
SLIPPAGE_BPS = float(os.getenv("SLIPPAGE_BPS", "0.0008"))
MAX_LIVE_DRAWDOWN = float(os.getenv("MAX_LIVE_DRAWDOWN", "0.2"))
ALLOW_SHORT = os.getenv("ALLOW_SHORT", "false").lower() == "true"

# Optional vendors (unused in this baseline but kept for future)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")
FINNHUB_API_KEY  = os.getenv("FINNHUB_API_KEY", "")
NEWSAPI_KEY      = os.getenv("NEWSAPI_KEY", "")

# Feature flags
INCLUDE_TECH_FEATURES = os.getenv("INCLUDE_TECH_FEATURES", "true").lower() == "true"
INCLUDE_FUNDAMENTALS  = os.getenv("INCLUDE_FUNDAMENTALS", "false").lower() == "true"
INCLUDE_NEWS          = os.getenv("INCLUDE_NEWS", "false").lower() == "true"

# Ensure folders exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# A lightweight lock to coordinate save/load of checkpoints across threads
AGENT_LOCK = threading.Lock()

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
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"}
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
        try:
            df = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv", timeout=12)
            col = "Symbol" if "Symbol" in df.columns else "symbol"
            syms = df[col].astype(str).str.replace(".", "-", regex=False).str.strip().tolist()
            syms = [s.replace("BRK.B", "BRK-B").replace("BF.B", "BF-B") for s in syms]
            seen, out = set(), []
            for s in syms:
                if s and s not in seen:
                    seen.add(s); out.append(s)
            if len(out) >= 50:
                tickers = out
        except Exception:
            pass
    if not tickers:
        try:
            yf_list = getattr(yf, "tickers_sp500", None)
            if callable(yf_list):
                out = [s.replace(".", "-") for s in yf.tickers_sp500()]
                if len(out) >= 50:
                    tickers = out
        except Exception:
            pass
    if not tickers:
        tickers = FALLBACK_TICKERS
    _save_sp500_cache(tickers)
    return tickers

# -------- Feature engineering helpers --------
def _ema_ratio(closes: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.DataFrame:
    e1 = closes.ewm(span=fast, adjust=False).mean()
    e2 = closes.ewm(span=slow, adjust=False).mean()
    return (e1 / (e2 + 1e-12)) - 1.0

def _rsi(closes: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = closes.diff()
    gain = (delta.clip(lower=0)).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    return 1 - (1 / (1 + rs))  # 0..1

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

# Generic chunked downloader for daily bars
def _download_daily_chunked(tickers: List[str], start: datetime, end: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames_p, frames_v = [], []
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i+100]
        dfi = yf.download(chunk, start=start, end=end, interval="1d", auto_adjust=True,
                          progress=False, group_by="column", threads=True)
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
    """Bounded, resilient fetch with simple tech features."""
    synthetic = False
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
    else:  # 1m intraday
        try:
            period = f"{max(5, min(60, int(intraday_days)))}d"
            dfi = yf.download(tickers[:min(MAX_TICKERS, 100)], period=period, interval="1m", auto_adjust=True,
                              progress=False, group_by="column", threads=True)
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
    synthetic = True
    idx = pd.date_range(end=now_utc().date(), periods=180, freq="D")
    synth = {}
    for s in (tickers[:20] or ["AAPL"]):
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(idx)))
        synth[s] = prices
    prices_df = pd.DataFrame(synth, index=idx)
    returns_df = prices_df.pct_change().dropna(how="all")
    return HistoryBundle(prices_df, returns_df, list(prices_df.columns), None, None, None, synthetic=True)

# ===================== ENV =====================
@dataclass
class EpState:
    t: int
    equity: float

class Env:
    """Environment with multi-channel features, strict causality.
       obs[t] -> W x (N * C + 1) where C is number of channels (returns + tech + slow features), +1 = risk column.
    """
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

        # Features per asset (align on returns index)
        idx = rets.index
        chans = [rets.values.astype(np.float32)]  # returns
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
        # Stack channels horizontally -> [T, N*C]
        self.X = np.concatenate(chans, axis=1)
        self.P = closes[self.tickers].reindex(idx).values.astype(np.float32)
        # Build rolling windows
        self.valid = list(range(self.W, self.X.shape[0]-1))
        self._obs_raw = self._build_obs_raw()  # [T-W, W, N*C]

        # episode-block sampling
        self.block_episodes = EPISODE_BLOCK
        self._episodes_in_block = 0
        self._block_risk = self._sample_risk()
        self._block_cash = self._sample_cash()

    def _build_obs_raw(self) -> np.ndarray:
        out = []
        for i in range(self.W, self.X.shape[0]):
            out.append(self.X[i-self.W:i, :])
        return np.array(out, dtype=np.float32)

    def _with_risk(self, raw: np.ndarray, risk: float) -> np.ndarray:
        risk_col = np.full((self.W, 1), float(risk), dtype=np.float32)
        return np.concatenate([raw, risk_col], axis=1)  # [W, N*C+1]

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
        realized_r = 0.0 if a == self.cash_idx else float(self.X[next_t, a % (self.N_assets)])  # first channel approx
        pnl = (s.equity * self.risk) * (realized_r - self.fee_bps)
        eq = s.equity + pnl
        rew = pnl / self.e0
        ns = EpState(t=next_t, equity=eq)
        done = ns.t >= (self.X.shape[0]-2)
        exec_price = float(self.P[next_t, a % (self.N_assets)]) if (a != self.cash_idx and self.N_assets>0) else float(self.P[next_t,0])
        return self._with_risk(self._obs_raw[ns.t-self.W], self.risk), rew, done, ns, realized_r, exec_price

# ===================== PPO =====================
class Net(nn.Module):
    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        self.nA = n_actions
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, self.nA)
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
        x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # [1, W, F]
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

PPO_CFG = {
    'lr':3e-4,'gamma':0.99,'gae_lambda':0.95,'clip_eps':0.2,
    'epochs':4,'minibatch_size':256,'entropy_coef':0.01,
    'value_coef':0.5,'max_grad_norm':0.5,'rollout_steps':2048
}

def infer_input_dim(env: Env) -> int:
    obs0, _ = env.reset(seed=0)           # [W, F]
    return int(obs0.shape[0] * obs0.shape[1])

# ===================== SAMPLERS =====================
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
}
STATE_LOCK = threading.Lock()
REINIT_EVENT = threading.Event()

TRAIN_STATS = {"last_reward_mean": 0.0, "last_win_rate": 0.0, "last_time_utc": None}

# ===================== PORTFOLIO =====================
class Portfolio:
    def __init__(self, cash: float):
        self.cash = float(cash)
        self.positions: Dict[str, float] = {}
        self.cost_basis: Dict[str, float] = {}
        self.equity = float(cash)
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0

    def mark_to_market(self, prices: Dict[str, float]):
        upnl = 0.0
        inv = 0.0
        for t, q in self.positions.items():
            p = prices.get(t, 0.0)
            cb = self.cost_basis.get(t, 0.0)
            upnl += q * (p - cb)
            inv += q * p
        self.unrealized_pnl = upnl
        self.equity = self.cash + inv + upnl - inv  # cash + (positions at CB) + upnl -> cash + MTM
        return self.equity

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        bps = SLIPPAGE_BPS
        return price * (1.0 + bps if is_buy else 1.0 - bps)

    def trade_notional(self, ticker: str, notional: float, price: float, fees_bps: float) -> Tuple[float, float]:
        if price <= 0:
            return 0.0, 0.0
        is_buy = notional > 0
        px = self._apply_slippage(price, is_buy)
        qty = notional / px
        fee = abs(notional) * fees_bps
        realized = 0.0
        prev_qty = self.positions.get(ticker, 0.0)
        prev_cb  = self.cost_basis.get(ticker, 0.0)
        if is_buy:
            new_qty = prev_qty + qty
            new_cb = (prev_cb * prev_qty + px * qty) / new_qty if new_qty != 0 else 0.0
            self.positions[ticker] = new_qty
            self.cost_basis[ticker] = new_cb
            self.cash -= (qty * px + fee)
        else:
            sell_qty = abs(qty)
            if ALLOW_SHORT or prev_qty > 0:
                exec_qty = sell_qty if ALLOW_SHORT else min(sell_qty, prev_qty)
                if prev_qty > 0 and exec_qty > 0:
                    realized = exec_qty * (px - prev_cb)
                self.positions[ticker] = prev_qty - exec_qty
                if self.positions[ticker] == 0:
                    self.cost_basis[ticker] = 0.0
                self.cash += (exec_qty * px) - fee
        self.realized_pnl += realized
        return (qty if is_buy else -min(abs(qty), prev_qty)), realized

# ===================== BUILD ENV =====================
def build_env() -> Tuple[Env, List[str], HistoryBundle]:
    tickers = list(FALLBACK_TICKERS)
    if AUTO_FETCH_SP500:
        try:
            sp = fetch_sp500()
            if sp and len(sp) >= 5:
                tickers = sp
        except Exception:
            pass
    bundle = fetch_history(tickers, TRAIN_INTERVAL, HISTORY_YEARS, INTRADAY_PERIOD_DAYS)
    env = Env(bundle, WINDOW, FEE_BPS)
    with STATE_LOCK:
        STATE["universe"] = env.N_assets
    return env, list(env.tickers), bundle

# ===================== ROLLOUT =====================
def rollout(env: Env, agent: PPO, steps: int):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [],[],[],[],[]
    obs, s = env.reset()
    for _ in range(steps):
        a, lp, v = agent.act(obs)
        nobs, r, done, s, _, _ = env.step(a, s)
        obs_buf.append(obs); act_buf.append(a); logp_buf.append(lp); rew_buf.append(r); val_buf.append(v)
        obs = nobs
        if done:
            obs, s = env.reset()
    gamma=PPO_CFG['gamma']; lam=PPO_CFG['gae_lambda']
    rewards=np.array(rew_buf,dtype=np.float32); values=np.array(val_buf+[0.0],dtype=np.float32)
    adv=np.zeros_like(rewards); gae=0.0
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*values[t+1]-values[t]
        gae=delta+gamma*lam*gae; adv[t]=gae
    ret=values[:-1]+adv
    adv=(adv-adv.mean())/(adv.std()+1e-8)
    batch={
        'obs':torch.tensor(np.array(obs_buf),dtype=torch.float32),
        'actions':torch.tensor(np.array(act_buf),dtype=torch.long),
        'logp':torch.tensor(np.array(logp_buf),dtype=torch.float32),
        'ret':torch.tensor(ret,dtype=torch.float32),
        'adv':torch.tensor(adv,dtype=torch.float32)
    }
    return batch

MODEL_PATH = os.path.join(MODEL_DIR, 'ppo.pt')

def save_checkpoint(agent: PPO, tag: str):
    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    ckpt = os.path.join(MODEL_DIR, f"ppo-{tag}-{ts}.pt")
    tmp  = os.path.join(MODEL_DIR, f"ppo-tmp.pt")
    payload = {"net": agent.net.state_dict(), "opt": agent.opt.state_dict(), "cfg": agent.cfg, "stats": TRAIN_STATS}
    with AGENT_LOCK:
        torch.save(payload, tmp)
        shutil.copy2(tmp, ckpt)
        shutil.copy2(tmp, MODEL_PATH)

# ===================== TRAINER (continuous) =====================
def trainer_thread(stop: threading.Event):
    env, _, bundle = build_env()
    input_dim  = infer_input_dim(env)
    n_actions  = env.N_assets + 1
    agent = PPO(input_dim, n_actions, PPO_CFG)

    # Warm load if exists (tolerate shape changes)
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location='cpu')
            if isinstance(state, dict) and 'net' in state:
                with AGENT_LOCK:
                    agent.net.load_state_dict(state['net'], strict=False)
                    if 'opt' in state:
                        try: agent.opt.load_state_dict(state['opt'])
                        except Exception: pass
        except Exception:
            pass

    while not stop.is_set():
        try:
            if not CONTINUOUS_TRAIN:
                time.sleep(5)
                continue
            # Small continuous updates
            batch = rollout(env, agent, min(CONTINUOUS_ROLLOUT_STEPS, PPO_CFG['rollout_steps']))
            agent.update(batch)
            # Track stats & save occasionally
            r = float(batch['ret'].mean().item())
            w = float((batch['ret']>0).float().mean().item())
            TRAIN_STATS.update({"last_reward_mean": r, "last_win_rate": w, "last_time_utc": now_utc().isoformat()})
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO train_events(ts_utc, interval, reward_mean, win_rate, notes) VALUES (?,?,?,?,?)",
                    (TRAIN_STATS['last_time_utc'], TRAIN_INTERVAL, r, w, "continuous"))
                conn.commit()
            save_checkpoint(agent, "cont")
            # Do not override UI mode; just keep stats
            time.sleep(CONTINUOUS_SLEEP_SECS)
        except Exception as e:
            with STATE_LOCK:
                STATE['status'] = f"train-error: {e}"
            time.sleep(5)

# ===================== LIVE / BACKTEST =====================
class PendingDecision:
    def __init__(self):
        self.action: Optional[int] = None
        self.risk: float = 0.0

def insert_trade(row: Dict):
    with get_db() as conn:
        cols = ','.join(row.keys()); ph = ','.join(['?']*len(row))
        conn.execute(f"INSERT INTO trades ({cols}) VALUES ({ph})", list(row.values()))
        conn.commit()

def live_thread(stop: threading.Event):
    env, tickers, bundle = build_env()
    input_dim  = infer_input_dim(env)
    n_actions  = env.N_assets + 1
    agent = PPO(input_dim, n_actions, PPO_CFG)

    # Attempt to load the latest (tolerate shape changes)
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location='cpu')
            if isinstance(state, dict) and 'net' in state:
                with AGENT_LOCK:
                    agent.net.load_state_dict(state['net'], strict=False)
        except Exception:
            pass

    tz = pytz.timezone(MARKET_TZ)
    today_et = now_utc().astimezone(tz).date()

    portfolio: Optional[Portfolio] = None
    day_risk: Optional[float] = None
    day_cash: Optional[float] = None
    live_session_id: Optional[str] = None

    last_prices: Optional[pd.Series] = None
    intraday_buf: List[np.ndarray] = []
    decision = PendingDecision()

    with STATE_LOCK:
        STATE.update({"today_trades": 0, "today_pnl": 0.0, "max_drawdown": 0.0})

    last_ckpt_check = 0.0
    last_loaded_mtime = 0.0

    def prices_to_map(series: pd.Series) -> Dict[str, float]:
        return {k: float(series.get(k, np.nan)) for k in tickers if k in series.index and not np.isnan(series[k])}

    def start_live_session():
        nonlocal day_risk, day_cash, live_session_id, portfolio, last_prices, intraday_buf
        day_risk = sample_risk(); day_cash = sample_cash()
        live_session_id = f"live-{today_et.isoformat()}"
        portfolio = Portfolio(cash=day_cash)
        last_prices = None; intraday_buf.clear()
        with STATE_LOCK:
            STATE['session_id'] = live_session_id
            STATE['equity'] = portfolio.equity
            STATE['cash'] = portfolio.cash
            STATE['positions_value'] = 0.0
            STATE['unrealized_pnl'] = 0.0
            STATE['today_pnl'] = 0.0
            STATE['today_trades'] = 0
        et = now_utc().astimezone(tz)
        insert_trade({
            "session_id": live_session_id, "mode": "LIVE",
            "ts_utc": now_utc().isoformat(), "ts_et": et.isoformat(),
            "side": "HOLD", "ticker": "CASH", "qty": 0.0, "fill_price": 0.0,
            "notional": 0.0, "fees_bps": FEE_BPS, "slippage_bps": SLIPPAGE_BPS,
            "risk_frac": float(day_risk), "position_before": 0.0, "position_after": 0.0,
            "cost_basis_before": 0.0, "cost_basis_after": 0.0,
            "realized_pnl": 0.0, "realized_pnl_pct": 0.0,
            "equity_after": portfolio.equity, "reason": "session start"
        })

    def flatten_all(reason: str):
        nonlocal portfolio
        if portfolio is None:
            return
        prices_map = {}
        if last_prices is not None:
            prices_map = prices_to_map(last_prices)
        for t, q in list(portfolio.positions.items()):
            p = prices_map.get(t, 0.0)
            if p <= 0:
                continue
            notional = -q * p
            qty_exec, realized = portfolio.trade_notional(t, notional, p, FEE_BPS)
            et = now_utc().astimezone(tz)
            insert_trade({
                "session_id": live_session_id, "mode": "LIVE",
                "ts_utc": now_utc().isoformat(), "ts_et": et.isoformat(),
                "side": "SELL" if q>0 else "BUY", "ticker": t, "qty": abs(qty_exec), "fill_price": p,
                "notional": abs(notional), "fees_bps": FEE_BPS, "slippage_bps": SLIPPAGE_BPS,
                "risk_frac": float(day_risk or 0.0), "position_before": float(q), "position_after": float(portfolio.positions.get(t,0.0)),
                "cost_basis_before": 0.0, "cost_basis_after": float(portfolio.cost_basis.get(t,0.0)),
                "realized_pnl": realized, "realized_pnl_pct": 0.0,
                "equity_after": portfolio.equity, "reason": reason
            })
        portfolio.mark_to_market(prices_map)

    def do_full_reinit():
        nonlocal env, tickers, input_dim, n_actions, agent
        nonlocal portfolio, day_risk, day_cash, live_session_id
        nonlocal last_prices, intraday_buf, decision, today_et
        env, tickers, _ = build_env()
        input_dim = infer_input_dim(env)
        n_actions = env.N_assets + 1
        agent = PPO(input_dim, n_actions, PPO_CFG)
        if os.path.exists(MODEL_PATH):
            try:
                state = torch.load(MODEL_PATH, map_location='cpu')
                if isinstance(state, dict) and 'net' in state:
                    with AGENT_LOCK:
                        agent.net.load_state_dict(state['net'], strict=False)
            except Exception:
                pass
        today_et = now_utc().astimezone(tz).date()
        portfolio = None; day_risk = day_cash = None; live_session_id = None
        last_prices = None; intraday_buf = []; decision = PendingDecision()
        with STATE_LOCK:
            STATE.update({
                "today_trades": 0, "today_pnl": 0.0, "max_drawdown": 0.0,
                "session_id": "bootstrap", "equity": 0.0, "cash": 0.0,
                "positions_value": 0.0, "unrealized_pnl": 0.0, "mode": "idle", "status": "idle"
            })

    while not stop.is_set():
        try:
            # Hot-reload newest checkpoint occasionally
            now_ts = time.time()
            if (now_ts - last_ckpt_check) >= CHECKPOINT_POLL_SECS:
                last_ckpt_check = now_ts
                try:
                    mtime = pathlib.Path(MODEL_PATH).stat().st_mtime if os.path.exists(MODEL_PATH) else 0
                    if mtime > last_loaded_mtime:
                        state = torch.load(MODEL_PATH, map_location='cpu')
                        if isinstance(state, dict) and 'net' in state:
                            with AGENT_LOCK:
                                agent.net.load_state_dict(state['net'], strict=False)
                            last_loaded_mtime = mtime
                except Exception:
                    pass

            if REINIT_EVENT.is_set():
                do_full_reinit(); REINIT_EVENT.clear(); time.sleep(0.5); continue

            et_now = now_utc().astimezone(tz)
            if et_now.date() != today_et:
                today_et = et_now.date(); day_risk = day_cash = None; portfolio = None; live_session_id = None; decision = PendingDecision()

            live_open = is_market_open(now_utc())

            if live_open:
                with STATE_LOCK: STATE['status'] = STATE['mode'] = 'live'
                if live_session_id is None:
                    start_live_session()

                # Drawdown gate
                if portfolio and portfolio.equity > 0:
                    peak = max(STATE.get('peak_equity', portfolio.equity), portfolio.equity)
                    with STATE_LOCK: STATE['peak_equity'] = peak
                    dd = (peak - portfolio.equity) / peak if peak > 0 else 0.0
                    if dd >= MAX_LIVE_DRAWDOWN:
                        flatten_all("max drawdown breach"); time.sleep(POLL_SECONDS); continue

                # Pull latest minute data
                live_names = tickers[:MAX_LIVE_TICKERS]
                data = yf.download(live_names, period="2d", interval="1m", progress=False, auto_adjust=True)
                closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
                latest = closes.iloc[-1].dropna()
                if last_prices is None:
                    last_prices = latest; time.sleep(POLL_SECONDS); continue

                # Execute previous decision
                if decision.action is not None and portfolio is not None:
                    a = int(decision.action); risk = float(decision.risk)
                    if a != env.cash_idx and a < len(live_names):
                        ticker = live_names[a]
                        price = float(latest.get(ticker, np.nan))
                        if not np.isnan(price) and price > 0:
                            notional = portfolio.equity * risk
                            qty_exec, realized = portfolio.trade_notional(ticker, notional, price, FEE_BPS)
                            et = now_utc().astimezone(tz)
                            insert_trade({
                                "session_id": live_session_id, "mode": "LIVE",
                                "ts_utc": now_utc().isoformat(), "ts_et": et.isoformat(),
                                "side": "BUY" if notional>0 else "SELL", "ticker": ticker,
                                "qty": abs(qty_exec), "fill_price": price, "notional": abs(notional),
                                "fees_bps": FEE_BPS, "slippage_bps": SLIPPAGE_BPS, "risk_frac": risk,
                                "position_before": 0.0, "position_after": float(portfolio.positions.get(ticker,0.0)),
                                "cost_basis_before": 0.0, "cost_basis_after": float(portfolio.cost_basis.get(ticker,0.0)),
                                "realized_pnl": realized, "realized_pnl_pct": 0.0,
                                "equity_after": portfolio.equity, "reason": "policy decision filled"
                            })
                            with STATE_LOCK: STATE['today_trades'] += 1

                # Update marks and state
                prices_map = {t: float(latest.get(t, np.nan)) for t in live_names if not np.isnan(latest.get(t, np.nan))}
                if portfolio is not None:
                    portfolio.mark_to_market(prices_map)
                    positions_value = sum((prices_map.get(t, 0.0) * q) for t, q in (portfolio.positions.items()))
                    with STATE_LOCK:
                        STATE['cash'] = float(portfolio.cash)
                        STATE['positions_value'] = float(positions_value)
                        STATE['unrealized_pnl'] = float(portfolio.unrealized_pnl)
                        STATE['equity'] = float(portfolio.equity)

                # Build obs for next decision
                common = latest.index.intersection(last_prices.index)
                rets = (latest[common] / last_prices[common] - 1.0).reindex(live_names).fillna(0.0)
                last_prices = latest
                intraday_buf.append(rets.values.astype(np.float32))
                if len(intraday_buf) > WINDOW: intraday_buf.pop(0)
                if len(intraday_buf) < WINDOW:
                    time.sleep(POLL_SECONDS); continue
                obs_np = np.stack(intraday_buf, axis=0).astype(np.float32)
                risk = float(day_risk or sample_risk())
                obs_np = np.concatenate([obs_np, np.full((WINDOW,1), risk, np.float32)], axis=1)
                with torch.no_grad():
                    logits, _ = agent.net(torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0))
                    a = int(torch.argmax(logits, dim=-1).item())
                decision.action = a; decision.risk = risk

                with STATE_LOCK:
                    peak_eq = max(STATE.get('peak_equity', portfolio.equity if portfolio else 0.0), portfolio.equity if portfolio else 0.0)
                    STATE['peak_equity'] = peak_eq
                    dd = 0.0 if peak_eq <= 0 else (peak_eq - (portfolio.equity if portfolio else 0.0)) / peak_eq
                    STATE['max_drawdown'] = float(dd)
                time.sleep(POLL_SECONDS); continue

            # Market closed: run light backtest loop to keep data + UI moving
            if BACKTEST_WHEN_CLOSED:
                with STATE_LOCK: STATE['status'] = STATE['mode'] = 'backtest'
                back_obs, back_s = env.reset(); backtest_session_id = f"bt-{int(time.time())}"; eq = float(env.e0)
                et = now_utc().astimezone(tz)
                insert_trade({
                    "session_id": backtest_session_id, "mode": "BACKTEST",
                    "ts_utc": now_utc().isoformat(), "ts_et": et.isoformat(),
                    "side": "HOLD", "ticker": "CASH", "qty": 0.0, "fill_price": 0.0,
                    "notional": 0.0, "fees_bps": FEE_BPS, "slippage_bps": SLIPPAGE_BPS,
                    "risk_frac": float(back_obs[0,-1]), "position_before": 0.0, "position_after": 0.0,
                    "cost_basis_before": 0.0, "cost_basis_after": 0.0,
                    "realized_pnl": 0.0, "realized_pnl_pct": 0.0, "equity_after": eq, "reason": "block start"
                })
                with STATE_LOCK:
                    STATE['session_id'] = backtest_session_id; STATE['equity'] = eq; STATE['today_trades'] = 0; STATE['today_pnl'] = 0.0
                for _ in range(2000):
                    a, lp, v = agent.act(back_obs)
                    nobs, r, done, back_s, realized_r, exec_price = env.step(a, back_s)
                    risk = float(back_obs[0,-1]); notional = eq * risk; fee = abs(notional) * FEE_BPS; pnl = notional * realized_r - fee; eq += pnl
                    ticker = env.tickers[a] if a != env.cash_idx and a < env.N_assets else "CASH"
                    qty = (abs(notional) / max(exec_price, 1e-6)) if ticker != "CASH" else 0.0
                    et = now_utc().astimezone(tz)
                    insert_trade({
                        "session_id": backtest_session_id, "mode": "BACKTEST",
                        "ts_utc": now_utc().isoformat(), "ts_et": et.isoformat(),
                        "side": "MARK", "ticker": ticker, "qty": qty, "fill_price": exec_price,
                        "notional": abs(notional), "fees_bps": FEE_BPS, "slippage_bps": SLIPPAGE_BPS,
                        "risk_frac": risk, "position_before": 0.0, "position_after": 0.0,
                        "cost_basis_before": 0.0, "cost_basis_after": 0.0,
                        "realized_pnl": pnl, "realized_pnl_pct": (pnl/abs(notional) if notional!=0 else 0.0),
                        "equity_after": eq, "reason": "backtest step"
                    })
                    with STATE_LOCK:
                        STATE['today_trades'] += 1; STATE['equity'] = float(eq); STATE['today_pnl'] = float(STATE.get('today_pnl', 0.0) + pnl)
                        peak_eq = max(STATE.get('peak_equity', eq), eq); STATE['peak_equity'] = peak_eq; STATE['max_drawdown'] = float(0.0 if peak_eq<=0 else (peak_eq - eq)/peak_eq)
                    back_obs = nobs
                    if done: break
                    time.sleep(BACKTEST_POLL_SECONDS)
                time.sleep(3); continue

            with STATE_LOCK: STATE['status'] = STATE['mode'] = 'idle'; time.sleep(5)
        except Exception as e:
            with STATE_LOCK: STATE['status'] = STATE['mode'] = f"error: {e}"
            time.sleep(POLL_SECONDS)

# ===================== API =====================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[CORS_ORIGIN] if CORS_ORIGIN != "*" else ["*"],
    allow_credentials=False, allow_methods=["*"], allow_headers=["*"]
)

def require_key(authorization: Optional[str] = Header(default=None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, detail="Missing bearer token")
    token = authorization.split()[1]
    if token != API_KEY:
        raise HTTPException(403, detail="Bad token")

@app.get("/healthz")
def healthz():
    return {"ok": True, "threads": {"trainer": True, "live": True}}

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

@app.get("/metrics")
def metrics(_: None = Depends(require_key)):
    with STATE_LOCK:
        return JSONResponse(STATE.copy())

@app.get("/stats")
def stats(_: None = Depends(require_key)):
    with STATE_LOCK:
        s = {k: STATE.get(k) for k in ("train_reward_mean","train_win_rate","last_train_utc","universe","max_drawdown","today_trades","cash","positions_value","unrealized_pnl","equity")}
    return {"train": TRAIN_STATS, "state": s}

class TradesQuery(BaseModel):
    limit: int = 100
    cursor: int = 0
    session: Optional[str] = None

@app.get("/trades")
def trades(limit: int = Query(100, ge=1, le=500), cursor: int = 0, session: Optional[str] = None, _: None = Depends(require_key)):
    if session is None:
        with STATE_LOCK:
            session = STATE.get("session_id", "bootstrap")
    with get_db() as conn:
        cur = conn.execute(
            "SELECT * FROM trades WHERE session_id=? AND trade_id>? ORDER BY trade_id DESC LIMIT ?",
            (session, cursor, limit),
        )
        rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    next_cursor = rows[-1]["trade_id"] if rows else cursor
    return {"data": rows, "next_cursor": next_cursor, "session": session}

@app.get("/equity")
def equity(window: str = "30d", session: Optional[str] = None, _: None = Depends(require_key)):
    if session is None:
        with STATE_LOCK:
            session = STATE.get("session_id", "bootstrap")
    with get_db() as conn:
        cur = conn.execute("SELECT ts_utc, equity_after FROM trades WHERE session_id=? ORDER BY trade_id ASC", (session,))
        rows = cur.fetchall()
    if rows:
        series = [{"t": r[0], "equity": float(r[1])} for r in rows if r[1] is not None]
        return {"series": series[-2000:], "session": session}
    with STATE_LOCK: eq = float(STATE.get("equity", 1000.0))
    now = now_utc()
    series = [{"t": (now - timedelta(minutes=m)).isoformat(), "equity": eq} for m in range(60, -1, -1)]
    return {"series": series, "session": session}

@app.post("/snapshot")
def snapshot(_: None = Depends(require_key)):
    ts = datetime.utcnow().strftime("%H%M%S")
    try:
        env, _, _ = build_env()
        input_dim = infer_input_dim(env)
        agent = PPO(input_dim, env.N_assets+1, PPO_CFG)
        if os.path.exists(MODEL_PATH):
            state = torch.load(MODEL_PATH, map_location='cpu')
            if isinstance(state, dict) and 'net' in state:
                with AGENT_LOCK:
                    agent.net.load_state_dict(state['net'], strict=False)
        save_checkpoint(agent, f"manual-{ts}")
        return {"ok": True}
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# ---------- Admin hard reset ----------
class ResetReq(BaseModel):
    password: str

def hard_reset():
    with STATE_LOCK:
        STATE['status'] = STATE['mode'] = 'resetting'
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
    except Exception as e:
        print("Model delete error:", e)
    try:
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
    except Exception as e:
        print("DB delete error:", e)
    with get_db() as conn:
        conn.execute(SCHEMA_SQL); conn.execute(TRAIN_SQL); conn.commit()
    REINIT_EVENT.set()
    with STATE_LOCK:
        STATE.update({
            "status": "idle",
            "mode": "idle",
            "equity": 0.0,
            "cash": 0.0,
            "positions_value": 0.0,
            "unrealized_pnl": 0.0,
            "today_pnl": 0.0,
            "today_trades": 0,
            "max_drawdown": 0.0,
            "session_id": "bootstrap",
            "peak_equity": 0.0
        })

@app.post("/admin/reset")
def admin_reset(req: ResetReq, _: None = Depends(require_key)):
    if req.password != API_KEY and req.password != os.getenv("ADMIN_RESET_PASSWORD", "please-change-me"):
        raise HTTPException(403, detail="Bad admin password")
    hard_reset()
    return {"ok": True, "message": "Reset complete"}

# ===================== BOOT =====================
STOP = threading.Event()
threading.Thread(target=trainer_thread, args=(STOP,), daemon=True).start()
threading.Thread(target=live_thread, args=(STOP,), daemon=True).start()

if __name__ == "__main__":
    if API_KEY == "changeme":
        print("WARNING: Set API_KEY env var before exposing this service.")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
