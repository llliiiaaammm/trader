import os, math, json, time, random, threading, sqlite3, shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from fastapi import FastAPI, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Response
from pydantic import BaseModel
import uvicorn

# ================ CONFIG =================
API_KEY = os.getenv("API_KEY", "changeme")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")

# Persist to Render disk (/data) so redeploys keep state
MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")
DB_PATH = os.getenv("DB_PATH", "/data/trades.sqlite3")

PORT = int(os.getenv("PORT", "8000"))
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")

# Live polling cadence and backtest cadence
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
BACKTEST_POLL_SECONDS = int(os.getenv("BACKTEST_POLL_SECONDS", "5"))
BACKTEST_WHEN_CLOSED = os.getenv("BACKTEST_WHEN_CLOSED", "true").lower() == "true"

# Training window (Eastern time)
NIGHTLY_TRAIN_ET = os.getenv("NIGHTLY_TRAIN_ET", "22:00")  # 10pm ET
TRAIN_ROUNDS_PER_NIGHT = int(os.getenv("TRAIN_ROUNDS_PER_NIGHT", "4"))
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "5"))
WINDOW = int(os.getenv("WINDOW", "30"))   # lookback window (minutes for live; days for backtest)
AUTO_FETCH_SP500 = os.getenv("AUTO_FETCH_SP500", "true").lower() == "true"

# Risk/cash sampling (held constant for blocks of episodes in BACKTEST)
EPISODE_BLOCK = int(os.getenv("EPISODE_BLOCK", "100"))
START_CASH_MIN = float(os.getenv("START_CASH_MIN", "1"))
START_CASH_MAX = float(os.getenv("START_CASH_MAX", "1000"))
RISK_MIN = float(os.getenv("RISK_MIN", "0.01"))
RISK_MAX = float(os.getenv("RISK_MAX", "1.0"))
RISK_JITTER = float(os.getenv("RISK_JITTER", "0.05"))

# Trading frictions & safety
FEE_BPS = float(os.getenv("FEE_BPS", "0.0005"))
MAX_LIVE_DRAWDOWN = float(os.getenv("MAX_LIVE_DRAWDOWN", "0.20"))

# Admin reset password
ADMIN_RESET_PASSWORD = os.getenv("ADMIN_RESET_PASSWORD", "please-change-me")

# Make sure folders exist
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ================ TIME HELPERS =================
def now_utc():
    return datetime.now(pytz.UTC)

def is_market_open(ts: datetime) -> bool:
    et = ts.astimezone(pytz.timezone(MARKET_TZ))
    if et.weekday() >= 5: return False
    o = et.replace(hour=9, minute=30, second=0, microsecond=0)
    c = et.replace(hour=16, minute=0, second=0, microsecond=0)
    return o <= et <= c

def is_nightly(ts: datetime) -> bool:
    et = ts.astimezone(pytz.timezone(MARKET_TZ))
    hh, mm = map(int, NIGHTLY_TRAIN_ET.split(":"))
    start = et.replace(hour=hh, minute=mm, second=0, microsecond=0)
    end = start + timedelta(hours=2)
    return start <= et <= end

# ================ DATA =================
FALLBACK_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK-B","JPM","XOM","UNH",
    "V","MA","HD","PG","AVGO","LLY","JNJ","TSLA","COST","MRK"
]

# ------------------ S&P500 fetch (resilient + cached) ------------------
import json, time, requests

SP500_CACHE = "/data/sp500.json"
SP500_TTL   = 24 * 3600  # seconds

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

def fetch_sp500() -> List[str]:
    # 0) cached?
    cached = _load_sp500_cache()
    if cached:
        return cached

    tickers: List[str] = []

    # 1) Wikipedia (robust headers)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Safari/537.36"}
        html = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers=headers, timeout=12
        ).text
        tables = pd.read_html(html)
        for t in tables:
            if "Symbol" in t.columns or "Ticker symbol" in t.columns:
                col = "Symbol" if "Symbol" in t.columns else "Ticker symbol"
                syms = (
                    t[col].astype(str)
                    .str.replace(".", "-", regex=False)  # BRK.B -> BRK-B
                    .str.strip()
                    .tolist()
                )
                # Normalize a couple known dots:
                syms = [s.replace("BRK.B", "BRK-B").replace("BF.B", "BF-B") for s in syms]
                seen, out = set(), []
                for s in syms:
                    if s and s not in seen:
                        seen.add(s); out.append(s)
                if len(out) >= 50:  # sanity threshold
                    tickers = out
                    break
    except Exception:
        pass

    # 2) CSV mirror (DataHub)
    if not tickers:
        try:
            df = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv", timeout=12)
            col = "Symbol" if "Symbol" in df.columns else "symbol"
            syms = (
                df[col].astype(str)
                .str.replace(".", "-", regex=False)
                .str.strip()
                .tolist()
            )
            syms = [s.replace("BRK.B", "BRK-B").replace("BF.B", "BF-B") for s in syms]
            seen, out = set(), []
            for s in syms:
                if s and s not in seen:
                    seen.add(s); out.append(s)
            if len(out) >= 50:
                tickers = out
        except Exception:
            pass

    # 3) yfinance helper (if present)
    if not tickers:
        try:
            # yfinance exposes this on some versions
            yf_list = getattr(yf, "tickers_sp500", None)
            if callable(yf_list):
                out = [s.replace(".", "-") for s in yf.tickers_sp500()]
                if len(out) >= 50:
                    tickers = out
        except Exception:
            pass

    # 4) absolute fallback
    if not tickers:
        tickers = FALLBACK_TICKERS

    _save_sp500_cache(tickers)
    return tickers

def fetch_history(tickers: List[str], years: int) -> pd.DataFrame:
    """
    Fast, bounded history fetch.
    - Caps years to <=3
    - Limits tickers to first 50 (yfinance gets slow with hundreds)
    - Uses 1D daily bars
    - Falls back to a small synthetic dataset if Yahoo is slow/unavailable
    """
    years = max(1, min(int(years), 3))
    use = list(tickers)[:50] if tickers else list(FALLBACK_TICKERS)
    end = now_utc().date()
    start = end - timedelta(days=365 * years)

    try:
        df = yf.download(
            use,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )["Close"]

        if isinstance(df, pd.Series):
            df = df.to_frame()

        df = df.dropna(how="all")
        if not df.empty:
            return df
    except Exception:
        pass  # fall through to synthetic fallback

    # --- Synthetic fallback so the API responds immediately ---
    idx = pd.date_range(end=end, periods=180, freq="D")
    synth = {}
    # 10–20 synthetic tickers is enough for the agent to run
    for s in (use[:20] or ["AAPL"]):
        # random walk around 100
        prices = 100 + np.cumsum(np.random.normal(0, 1, len(idx)))
        synth[s] = prices
    return pd.DataFrame(synth, index=idx)

# ================ ENV =================
@dataclass
class EpState:
    t: int
    equity: float

class Env:
    def __init__(self, returns_df: pd.DataFrame, prices_df: pd.DataFrame, window: int, fee_bps: float):
        self.R = returns_df.dropna(axis=1, how='all').values.astype(np.float32)  # [T, N]
        self.P = prices_df[self.tickers].values.astype(np.float32)               # [T, N]  <-- NEW
        self.tickers = list(returns_df.columns)
        self.N_assets = len(self.tickers)
        self.W = window
        self.cash = self.N_assets
        self.fee_bps = fee_bps
        self._obs_raw = self._build_obs_raw()
        self.valid = list(range(self.W, self.R.shape[0]-1))
        # episode-block sampling...
        self.block_episodes = EPISODE_BLOCK
        self._episodes_in_block = 0
        self._block_risk = self._sample_risk()
        self._block_cash = self._sample_cash()

    def _build_obs_raw(self):
        out = []
        for i in range(self.W, self.R.shape[0]):
            out.append(self.R[i-self.W:i, :])  # [W, N_assets]
        return np.array(out, dtype=np.float32)

    def _with_risk(self, raw, risk: float):
        risk_col = np.full((self.W, 1), float(risk), dtype=np.float32)
        return np.concatenate([raw, risk_col], axis=1)  # [W, N_assets+1]

    def _sample_risk(self):
        base = random.random()
        val = RISK_MIN + (RISK_MAX - RISK_MIN) * (base ** 0.75)
        val *= (1.0 + random.uniform(-RISK_JITTER, RISK_JITTER))
        return float(max(RISK_MIN, min(RISK_MAX, val)))

    def _sample_cash(self):
        log_min, log_max = math.log(START_CASH_MIN), math.log(START_CASH_MAX)
        u = random.random()
        return float(math.exp(log_min + u * (log_max - log_min)))

    def _maybe_roll_block(self):
        if self._episodes_in_block >= self.block_episodes:
            self._episodes_in_block = 0
            self._block_risk = self._sample_risk()
            self._block_cash = self._sample_cash()

    def reset(self, starting_cash: Optional[float] = None, risk: Optional[float] = None, seed=None):
        if seed is not None:
            random.seed(seed); np.random.seed(seed)
        self._maybe_roll_block()
        self.e0 = float(self._block_cash if starting_cash is None else starting_cash)
        self.risk = float(self._block_risk if risk is None else risk)
        t = random.choice(self.valid)
        raw = self._obs_raw[t-self.W]
        self._episodes_in_block += 1
        return self._with_risk(raw, self.risk), EpState(t=t, equity=self.e0)

    def step(self, a: int, s: EpState):
        next_t = s.t + 1
        if next_t >= self.R.shape[0]-1:
            return self._with_risk(self._obs_raw[s.t-self.W], self.risk), 0.0, True, s
        r = 0.0 if a == self.cash else float(self.R[next_t, a])
        pnl = (s.equity * self.risk) * (r - self.fee_bps)
        eq = s.equity + pnl
        rew = pnl / self.e0
        ns = EpState(t=next_t, equity=eq)
        done = ns.t >= self.R.shape[0]-2
        return self._with_risk(self._obs_raw[ns.t-self.W], self.risk), rew, done, ns

# ================ PPO =================
class Net(nn.Module):
    def __init__(self, window: int, n_inputs: int, n_actions: int):
        super().__init__()
        self.nA = n_actions
        self.body = nn.Sequential(
            nn.Flatten(),
            nn.Linear(window * n_inputs, 256),
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
    def __init__(self, window, n_inputs, n_actions, cfg):
        self.net = Net(window, n_inputs, n_actions)
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

PPO_CFG = {
    'lr':3e-4,'gamma':0.99,'gae_lambda':0.95,'clip_eps':0.2,
    'epochs':4,'minibatch_size':256,'entropy_coef':0.01,
    'value_coef':0.5,'max_grad_norm':0.5,'rollout_steps':1024
}

# ================ SAMPLERS =================
def sample_risk():
    base = random.random()
    val = RISK_MIN + (RISK_MAX - RISK_MIN) * (base ** 0.75)
    val *= 1.0 + random.uniform(-RISK_JITTER, RISK_JITTER)
    return float(max(RISK_MIN, min(RISK_MAX, val)))

def sample_cash():
    log_min, log_max = math.log(START_CASH_MIN), math.log(START_CASH_MAX)
    u = random.random()
    return float(math.exp(log_min + u * (log_max - log_min)))

# ================ DB (SQLite) =================
def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
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
    )
    """)
    conn.commit()
    return conn

DB = db()

def insert_trade(row: Dict):
    cols = ','.join(row.keys()); ph = ','.join(['?']*len(row))
    DB.execute(f"INSERT INTO trades ({cols}) VALUES ({ph})", list(row.values()))
    DB.commit()

# ================ GLOBAL STATE =================
STATE = {
    "status": "idle",
    "equity": 0.0,
    "today_pnl": 0.0,
    "today_trades": 0,
    "universe": 0,
    "max_drawdown": 0.0,
    "mode": "idle",
    "session_id": "bootstrap"
}
STATE_LOCK = threading.Lock()

# **NEW**: reinit trigger used by /admin/reset
REINIT_EVENT = threading.Event()

# ================ BUILD ENV =================
def build_env():
    """
    Builds the environment quickly and always returns something usable.
    - Tries S&P500 list if enabled
    - Uses fast/bounded history
    - Guarantees non-empty returns (synthetic fallback if needed)
    """
    tickers = list(FALLBACK_TICKERS)
    if AUTO_FETCH_SP500:
        try:
            sp = fetch_sp500()
            if sp and len(sp) >= 5:
                tickers = sp
        except Exception:
            pass  # keep fallback list

    prices = fetch_history(tickers, HISTORY_YEARS)
    rets = prices.pct_change().dropna(how="all")

    # Guarantee at least one column of returns so the agent can start
    if rets.shape[1] == 0:
        idx = prices.index[1:] if len(prices.index) > 1 else pd.date_range(end=now_utc().date(), periods=120, freq="D")
        rets = pd.DataFrame({"AAPL": np.random.normal(0, 0.01, len(idx))}, index=idx)

    env = Env(rets, WINDOW, FEE_BPS)
    with STATE_LOCK:
        STATE["universe"] = env.N_assets
    return env, list(rets.columns)

# ================ TRAINER =================
def rollout(env: Env, agent: PPO, steps: int):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [],[],[],[],[]
    obs, s = env.reset()
    for _ in range(steps):
        a, lp, v = agent.act(obs)
        nobs, r, done, s = env.step(a, s)
        obs_buf.append(obs); act_buf.append(a); logp_buf.append(lp); rew_buf.append(r); val_buf.append(v)
        obs = nobs
        if done: obs, s = env.reset()
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
    torch.save(agent.net.state_dict(), tmp)
    shutil.copy2(tmp, ckpt)
    shutil.copy2(tmp, MODEL_PATH)

def trainer_thread(stop: threading.Event):
    env, _ = build_env()
    n_inputs  = env.N_assets + 1
    n_actions = env.N_assets + 1
    agent = PPO(WINDOW, n_inputs, n_actions, PPO_CFG)
    if os.path.exists(MODEL_PATH):
        agent.net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    while not stop.is_set():
        time.sleep(5)
        if is_market_open(now_utc()) or not is_nightly(now_utc()):
            continue
        with STATE_LOCK: STATE['status']=STATE['mode']='training'
        for _ in range(TRAIN_ROUNDS_PER_NIGHT):
            batch=rollout(env,agent,PPO_CFG['rollout_steps']); agent.update(batch)
        save_checkpoint(agent, "nightly")
        with STATE_LOCK: STATE['status']=STATE['mode']='idle'

# ================ LIVE / BACKTEST ENGINE =================
def live_thread(stop: threading.Event):
    env, tickers = build_env()
    n_inputs  = env.N_assets + 1
    n_actions = env.N_assets + 1
    agent = PPO(WINDOW, n_inputs, n_actions, PPO_CFG)
    if os.path.exists(MODEL_PATH):
        agent.net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    equity = 0.0
    peak_equity = 0.0
    today_pnl = 0.0

    tz = pytz.timezone(MARKET_TZ)
    today_et = now_utc().astimezone(tz).date()

    day_risk: Optional[float] = None
    day_cash: Optional[float] = None
    live_session_id: Optional[str] = None

    back_s = None
    back_obs = None
    backtest_session_id: Optional[str] = None
    last_block_counter: Optional[int] = None

    last_prices = None
    intraday_buf: List[np.ndarray] = []

    with STATE_LOCK:
        STATE.update({"today_trades": 0, "today_pnl": 0.0, "max_drawdown": 0.0})

    def log_trade_row(session_id, mode, side, ticker, qty, price, notional, fee_bps, risk, realized, realized_pct, equity_after, reason):
        et = now_utc().astimezone(tz)
        insert_trade({
            "session_id": session_id, "mode": mode,
            "ts_utc": now_utc().isoformat(), "ts_et": et.isoformat(),
            "side": side, "ticker": ticker, "qty": qty, "fill_price": price,
            "notional": notional, "fees_bps": fee_bps, "slippage_bps": 0.0,
            "risk_frac": risk, "position_before": 0.0, "position_after": qty,
            "cost_basis_before": 0.0, "cost_basis_after": price,
            "realized_pnl": realized, "realized_pnl_pct": realized_pct,
            "equity_after": equity_after, "reason": reason
        })

    def start_live_session():
        nonlocal day_risk, day_cash, live_session_id, equity, peak_equity, today_pnl, last_prices, intraday_buf
        day_risk = sample_risk()
        day_cash = sample_cash()
        live_session_id = f"live-{today_et.isoformat()}"
        equity = float(day_cash)
        peak_equity = equity
        today_pnl = 0.0
        last_prices = None
        intraday_buf.clear()
        with STATE_LOCK:
            STATE['session_id'] = live_session_id
            STATE['equity'] = equity
            STATE['today_pnl'] = 0.0
            STATE['today_trades'] = 0
        log_trade_row(live_session_id, "LIVE", "HOLD", "CASH", 0.0, 0.0, 0.0, FEE_BPS, day_risk, 0.0, 0.0, equity, "session start")

    # helper to rebuild runtime after reset
    def do_full_reinit():
        nonlocal env, tickers, n_inputs, n_actions, agent
        nonlocal equity, peak_equity, today_pnl, today_et
        nonlocal day_risk, day_cash, live_session_id
        nonlocal back_s, back_obs, backtest_session_id, last_block_counter
        nonlocal last_prices, intraday_buf

        env, tickers = build_env()
        n_inputs  = env.N_assets + 1
        n_actions = env.N_assets + 1
        agent = PPO(WINDOW, n_inputs, n_actions, PPO_CFG)
        if os.path.exists(MODEL_PATH):
            agent.net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

        equity = 0.0
        peak_equity = 0.0
        today_pnl = 0.0
        today_et = now_utc().astimezone(tz).date()

        day_risk = None
        day_cash = None
        live_session_id = None

        back_s = None
        back_obs = None
        backtest_session_id = None
        last_block_counter = None

        last_prices = None
        intraday_buf = []

        with STATE_LOCK:
            STATE.update({
                "today_trades": 0,
                "today_pnl": 0.0,
                "max_drawdown": 0.0,
                "session_id": "bootstrap",
                "equity": 0.0,
                "mode": "idle",
                "status": "idle"
            })

    while not stop.is_set():
        try:
            # react to admin reset
            if REINIT_EVENT.is_set():
                do_full_reinit()
                REINIT_EVENT.clear()
                time.sleep(0.5)
                continue

            et_now = now_utc().astimezone(tz)
            if et_now.date() != today_et:
                today_et = et_now.date()
                day_risk = day_cash = live_session_id = None

            live_open = is_market_open(now_utc())

            if live_open:
                with STATE_LOCK:
                    STATE['status'] = STATE['mode'] = 'live'
                if live_session_id is None:
                    start_live_session()

                dd = 0.0 if peak_equity <= 0 else (peak_equity - equity) / peak_equity
                if dd >= MAX_LIVE_DRAWDOWN:
                    time.sleep(POLL_SECONDS); continue

                data = yf.download(tickers, period="2d", interval="1m", progress=False, auto_adjust=True)
                closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
                latest = closes.iloc[-1].dropna()
                if last_prices is None:
                    last_prices = latest
                    time.sleep(POLL_SECONDS); continue

                common = latest.index.intersection(last_prices.index)
                rets = (latest[common] / last_prices[common] - 1.0)
                last_prices = latest

                intraday_buf.append(rets.reindex(tickers).fillna(0.0).values)
                if len(intraday_buf) > WINDOW: intraday_buf.pop(0)
                if len(intraday_buf) < WINDOW: time.sleep(POLL_SECONDS); continue

                obs_np = np.stack(intraday_buf, axis=0).astype(np.float32)
                risk = float(day_risk)
                obs_np = np.concatenate([obs_np, np.full((WINDOW,1), risk, np.float32)], axis=1)

                with torch.no_grad():
                    logits, _ = agent.net(torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0))
                    a = int(torch.argmax(logits, dim=-1).item())

                pnl = 0.0; price = float(latest.mean()) if len(latest)>0 else 0.0
                side = 'HOLD'; ticker='CASH'; qty=0.0
                if a != env.cash:
                    r = float(obs_np[-1][a])
                    ticker = tickers[a]
                    notional = equity * risk
                    qty = notional / max(price, 1e-6)
                    fee = notional * FEE_BPS
                    pnl = notional * r - fee
                    equity += pnl
                    peak_equity = max(peak_equity, equity)
                    side = 'BUY' if r >= 0 else 'SELL'
                    realized = pnl if side == 'SELL' else 0.0
                    realized_pct = (realized/notional) if notional>0 else 0.0
                    log_trade_row(live_session_id, "LIVE", side, ticker, qty, price, notional, FEE_BPS, risk, realized, realized_pct, equity, "policy argmax")
                    with STATE_LOCK: STATE['today_trades'] += 1

                today_pnl += pnl
                dd = 0.0 if peak_equity <= 0 else (peak_equity - equity) / peak_equity
                with STATE_LOCK:
                    STATE['equity'] = float(equity)
                    STATE['today_pnl'] = float(today_pnl)
                    STATE['max_drawdown'] = float(dd)
                time.sleep(POLL_SECONDS)
                continue

            if BACKTEST_WHEN_CLOSED:
                with STATE_LOCK:
                    STATE['status'] = STATE['mode'] = 'backtest'
                if back_s is None:
                    back_obs, back_s = env.reset()
                    backtest_session_id = f"bt-{int(time.time())}"
                    last_block_counter = env._episodes_in_block
                    equity = float(env.e0)
                    peak_equity = equity
                    today_pnl = 0.0
                    with STATE_LOCK:
                        STATE['session_id'] = backtest_session_id
                        STATE['equity'] = equity
                        STATE['today_pnl'] = 0.0
                        STATE['today_trades'] = 0
                    log_trade_row(
                        backtest_session_id, "BACKTEST", "HOLD", "CASH",
                        0.0, 0.0, 0.0, FEE_BPS, float(back_obs[0, -1]),
                        0.0, 0.0, equity, "block start"
                    )
            
                # record current step index before acting (to fetch the matching price)
                cur_t = back_s.t
                a, lp, v = agent.act(back_obs)
            
                # real historical price for this asset at cur_t (if not CASH)
                price = 0.0
                if a != env.cash and a < env.N_assets:
                    try:
                        price = float(env.P[cur_t, a])
                    except Exception:
                        price = 0.0
            
                nobs, r, done, back_s = env.step(a, back_s)
            
                if env._episodes_in_block == 1 and last_block_counter != 1:
                    backtest_session_id = f"bt-{int(time.time())}"
                    equity = float(env.e0)
                    peak_equity = equity
                    today_pnl = 0.0
                    with STATE_LOCK:
                        STATE['session_id'] = backtest_session_id
                        STATE['equity'] = equity
                        STATE['today_pnl'] = 0.0
                        STATE['today_trades'] = 0
                    log_trade_row(
                        backtest_session_id, "BACKTEST", "HOLD", "CASH",
                        0.0, 0.0, 0.0, FEE_BPS, float(nobs[0, -1]),
                        0.0, 0.0, equity, "block start"
                    )
                last_block_counter = env._episodes_in_block
            
                risk = float(back_obs[0, -1])
                notional = equity * risk
                fee = notional * FEE_BPS
                pnl = notional * r - fee
                equity += pnl
                peak_equity = max(peak_equity, equity)
                side = 'BUY' if r >= 0 else 'SELL'
                realized = pnl if side == 'SELL' else 0.0
                realized_pct = (realized / notional) if notional > 0 else 0.0
                ticker = tickers[a] if a != env.cash and a < len(tickers) else "CASH"
                qty = (notional / max(price, 1e-6)) if (a != env.cash and price > 0) else 0.0
            
                log_trade_row(
                    backtest_session_id, "BACKTEST", side, ticker, qty, price,
                    notional, FEE_BPS, risk, realized, realized_pct, equity, "backtest step"
                )
                with STATE_LOCK:
                    STATE['today_trades'] += 1
                    STATE['equity'] = float(equity)
                    STATE['today_pnl'] = float(STATE.get('today_pnl', 0.0) + pnl)
                    STATE['max_drawdown'] = float(0.0 if peak_equity <= 0 else (peak_equity - equity) / peak_equity)
            
                back_obs = nobs
                if done:
                    back_obs, back_s = env.reset()
                    last_block_counter = env._episodes_in_block
                time.sleep(BACKTEST_POLL_SECONDS)
                continue

            with STATE_LOCK:
                STATE['status'] = STATE['mode'] = 'idle'
            time.sleep(5)

        except Exception as e:
            with STATE_LOCK:
                STATE['status'] = STATE['mode'] = f"error: {e}"
            time.sleep(POLL_SECONDS)

# ================ API =================
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
    if token != API_KEY: raise HTTPException(403, detail="Bad token")

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

@app.get("/metrics")
def metrics(_: None = Depends(require_key)):
    with STATE_LOCK:
        return JSONResponse(STATE.copy())

@app.get("/trades")
def trades(limit: int = 100, cursor: int = 0, session: Optional[str] = None, _: None = Depends(require_key)):
    if session is None:
        with STATE_LOCK:
            session = STATE.get("session_id", "bootstrap")
    limit = max(1, min(500, limit))
    cur = DB.execute(
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
    cur = DB.execute("SELECT ts_utc, equity_after FROM trades WHERE session_id=? ORDER BY trade_id ASC", (session,))
    rows = cur.fetchall()
    if rows:
        series = [{"t": r[0], "equity": float(r[1])} for r in rows if r[1] is not None]
        return {"series": series[-2000:], "session": session}
    with STATE_LOCK: eq = float(STATE.get("equity", 1000.0))
    now = now_utc()
    series = [{"t": (now - timedelta(minutes=m)).isoformat(), "equity": eq} for m in range(60, -1, -1)]
    return {"series": series, "session": session}

@app.post("/seed")
def seed(_: None = Depends(require_key)):
    with STATE_LOCK:
        eq = float(STATE.get("equity", 1000.0))
        session_id = STATE.get("session_id", "seed")
    et = now_utc().astimezone(pytz.timezone(MARKET_TZ))
    for i in range(20):
        pnl = (i - 10) * 0.5
        eq += pnl
        insert_trade({
            "session_id": session_id,"mode":"BACKTEST","ts_utc":now_utc().isoformat(),"ts_et":et.isoformat(),
            "side":"SELL" if pnl>0 else "BUY","ticker":"AAPL","qty":1.0,"fill_price":100.0+i,"notional":100.0+i,
            "fees_bps":FEE_BPS,"slippage_bps":0.0,"risk_frac":0.1,"position_before":0.0,"position_after":1.0,
            "cost_basis_before":0.0,"cost_basis_after":100.0,"realized_pnl":pnl if pnl>0 else 0.0,
            "realized_pnl_pct":(pnl/(100.0+i)) if pnl>0 else 0.0,"equity_after":eq,"reason":"seed"
        })
    return {"ok": True}

@app.post("/snapshot")
def snapshot(_: None = Depends(require_key)):
    env, _ = build_env()
    n_inputs  = env.N_assets + 1
    n_actions = env.N_assets + 1
    agent = PPO(WINDOW, n_inputs, n_actions, PPO_CFG)
    if os.path.exists(MODEL_PATH):
        agent.net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    ts = datetime.utcnow().strftime("%H%M%S")
    save_checkpoint(agent, f"manual-{ts}")
    return {"ok": True}

# ---------- Admin hard reset ----------
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
    # recreate empty DB
    global DB
    DB = db()
    with STATE_LOCK:
        STATE.update({
            "status": "idle",
            "mode": "idle",
            "equity": 0.0,
            "today_pnl": 0.0,
            "today_trades": 0,
            "max_drawdown": 0.0,
            "session_id": "bootstrap"
        })
    # signal worker to rebuild env/agent & sessions
    REINIT_EVENT.set()

class ResetReq(BaseModel):
    password: str

@app.post("/admin/reset")
def admin_reset(req: ResetReq, _: None = Depends(require_key)):
    if req.password != ADMIN_RESET_PASSWORD:
        raise HTTPException(403, detail="Bad admin password")
    hard_reset()
    return {"ok": True, "message": "Reset complete"}

# ================ BOOT =================
STOP = threading.Event()
threading.Thread(target=trainer_thread, args=(STOP,), daemon=True).start()
threading.Thread(target=live_thread, args=(STOP,), daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
