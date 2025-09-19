import os, math, json, time, random, threading, sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

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
import uvicorn

# ------------------ CONFIG ------------------
API_KEY = os.getenv("API_KEY", "changeme")
CORS_ORIGIN = os.getenv("CORS_ORIGIN", "*")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
DB_PATH = os.getenv("DB_PATH", "data/trades.sqlite3")
PORT = int(os.getenv("PORT", "8000"))
MARKET_TZ = os.getenv("MARKET_TZ", "America/New_York")
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
AUTO_FETCH_SP500 = os.getenv("AUTO_FETCH_SP500", "true").lower() == "true"

START_CASH_MIN = float(os.getenv("START_CASH_MIN", "10"))
START_CASH_MAX = float(os.getenv("START_CASH_MAX", "10000"))
RISK_MIN = float(os.getenv("RISK_MIN", "0.02"))
RISK_MAX = float(os.getenv("RISK_MAX", "1.0"))
RISK_JITTER = float(os.getenv("RISK_JITTER", "0.05"))
FEE_BPS = float(os.getenv("FEE_BPS", "0.0005"))

HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "5"))
WINDOW = int(os.getenv("WINDOW", "30"))
NIGHTLY_TRAIN_ET = os.getenv("NIGHTLY_TRAIN_ET", "22:00")
TRAIN_ROUNDS_PER_NIGHT = int(os.getenv("TRAIN_ROUNDS_PER_NIGHT", "4"))

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------ TIME HELPERS ------------------
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

# ------------------ DATA ------------------
FALLBACK_TICKERS = [
    "AAPL","MSFT","AMZN","GOOGL","META","NVDA","BRK-B","JPM","XOM","UNH",
    "V","MA","HD","PG","AVGO","LLY","JNJ","TSLA","COST","MRK"
]
def fetch_sp500() -> List[str]:
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        for t in tables:
            if 'Symbol' in t.columns or 'Ticker symbol' in t.columns:
                col = 'Symbol' if 'Symbol' in t.columns else 'Ticker symbol'
                syms = t[col].astype(str).str.replace('.', '-', regex=False).str.strip().tolist()
                syms = [s.replace('BRK.B','BRK-B').replace('BF.B','BF-B') for s in syms]
                seen, out = set(), []
                for s in syms:
                    if s not in seen:
                        seen.add(s); out.append(s)
                return out
    except Exception:
        pass
    return FALLBACK_TICKERS

def fetch_history(tickers: List[str], years: int) -> pd.DataFrame:
    end = now_utc(); start = end - timedelta(days=int(years*365.25))
    df = yf.download(tickers, start=start.date(), end=end.date(), progress=False, auto_adjust=True)["Close"]
    if isinstance(df, pd.Series): df = df.to_frame()
    return df.dropna(how="all")

# ------------------ ENV ------------------
@dataclass
class EpState: 
    t: int
    equity: float

class Env:
    def __init__(self, returns_df: pd.DataFrame, window: int, fee_bps: float):
        self.R = returns_df.dropna(axis=1, how='all').values  # [T,N]
        self.tickers = list(returns_df.columns)
        self.N = len(self.tickers)
        self.W = window
        self.cash = self.N
        self.fee_bps = fee_bps
        self.obs = self._build_obs()
        self.valid = list(range(self.W, self.R.shape[0]-1))
    def _build_obs(self):
        out = []
        for i in range(self.W, self.R.shape[0]): out.append(self.R[i-self.W:i,:])
        return np.array(out)
    def reset(self, starting_cash: float, risk: float, seed=None):
        if seed is not None: random.seed(seed); np.random.seed(seed)
        t = random.choice(self.valid); self.e0 = starting_cash; self.risk = risk
        return self.obs[t-self.W], EpState(t=t, equity=starting_cash)
    def step(self, a: int, s: EpState):
        next_t = s.t + 1
        if next_t >= self.R.shape[0]-1:
            return self.obs[s.t-self.W], 0.0, True, s
        r = 0.0 if a==self.cash else float(self.R[next_t, a])
        pnl = (s.equity * self.risk) * (r - self.fee_bps)
        eq = s.equity + pnl
        rew = pnl / self.e0
        ns = EpState(t=next_t, equity=eq)
        return self.obs[ns.t-self.W], rew, ns.t >= self.R.shape[0]-2, ns

# ------------------ PPO ------------------
class Net(nn.Module):
    def __init__(self, window, n):
        super().__init__()
        self.nA = n+1
        self.body = nn.Sequential(nn.Flatten(), nn.Linear(window*n,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU())
        self.pi = nn.Linear(256, self.nA); self.v = nn.Linear(256,1)
    def forward(self,x): 
        z=self.body(x); 
        return self.pi(z), self.v(z)

class PPO:
    def __init__(self, window, n, cfg):
        self.net = Net(window,n); self.opt=optim.Adam(self.net.parameters(), lr=cfg['lr']); self.cfg=cfg
    @torch.no_grad()
    def act(self, obs):
        x=torch.tensor(obs,dtype=torch.float32).unsqueeze(0); logits,v=self.net(x)
        dist=torch.distributions.Categorical(logits=logits); a=dist.sample()
        return int(a.item()), dist.log_prob(a).item(), float(v.squeeze(0))
    def eval(self, obs, a):
        logits, v = self.net(obs); dist=torch.distributions.Categorical(logits=logits)
        return dist.log_prob(a), v.squeeze(-1), dist.entropy().mean()
    def update(self, batch):
        cfg=self.cfg; N=batch['obs'].size(0); idx=torch.randperm(N)
        for _ in range(cfg['epochs']):
            for i in range(0,N,cfg['minibatch_size']):
                mb=idx[i:i+cfg['minibatch_size']]
                obs=batch['obs'][mb]; a=batch['actions'][mb]; oldlp=batch['logp'][mb]
                adv=batch['adv'][mb]; ret=batch['ret'][mb]
                logp, v, ent = self.eval(obs,a)
                ratio=(logp-oldlp).exp(); s1=ratio*adv; s2=torch.clamp(ratio,1-cfg['clip_eps'],1+cfg['clip_eps'])*adv
                pol_loss=-torch.min(s1,s2).mean()
                v_loss=(ret-v).pow(2).mean()
                loss = pol_loss + cfg['value_coef']*v_loss - cfg['entropy_coef']*ent
                self.opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.net.parameters(), cfg['max_grad_norm']); self.opt.step()

PPO_CFG = {
    'lr':3e-4,'gamma':0.99,'gae_lambda':0.95,'clip_eps':0.2,'epochs':4,'minibatch_size':256,'entropy_coef':0.01,'value_coef':0.5,'max_grad_norm':0.5,'rollout_steps':1024
}

# ------------------ SAMPLING ------------------
def sample_risk():
    base=random.random(); val=RISK_MIN+(RISK_MAX-RISK_MIN)*(base**0.75); val*=1.0+random.uniform(-RISK_JITTER,RISK_JITTER)
    return float(max(RISK_MIN,min(RISK_MAX,val)))

def sample_cash():
    log_min,log_max=math.log(START_CASH_MIN),math.log(START_CASH_MAX); u=random.random()
    return float(math.exp(log_min+u*(log_max-log_min)))

# ------------------ DB (SQLite) ------------------
def db():
    conn=sqlite3.connect(DB_PATH, check_same_thread=False)
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
    conn.commit();
    return conn

DB = db()

def insert_trade(row: Dict):
    cols = ','.join(row.keys()); ph = ','.join(['?']*len(row))
    DB.execute(f"INSERT INTO trades ({cols}) VALUES ({ph})", list(row.values())); DB.commit()

# ------------------ GLOBAL STATE ------------------
STATE = {"status":"idle","equity":1000.0,"today_pnl":0.0,"today_trades":0,"universe":0,"max_drawdown":0.0}
STATE_LOCK = threading.Lock()

# ------------------ BUILD ENV ------------------
def build_env():
    tickers = fetch_sp500() if AUTO_FETCH_SP500 else FALLBACK_TICKERS
    prices = fetch_history(tickers, HISTORY_YEARS)
    rets = prices.pct_change().dropna(how='all')
    env = Env(rets, WINDOW, FEE_BPS)
    with STATE_LOCK: STATE['universe'] = env.N
    return env, tickers

# ------------------ TRAINER ------------------
def rollout(env: Env, agent: PPO, steps: int):
    obs_buf, act_buf, logp_buf, rew_buf, val_buf = [],[],[],[],[]
    obs, s = env.reset(sample_cash(), sample_risk())
    for _ in range(steps):
        a, lp, v = agent.act(obs)
        nobs, r, done, s = env.step(a, s)
        obs_buf.append(obs); act_buf.append(a); logp_buf.append(lp); rew_buf.append(r); val_buf.append(v)
        obs = nobs
        if done: obs, s = env.reset(sample_cash(), sample_risk())
    # GAE
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

def trainer_thread(stop: threading.Event):
    env, _ = build_env()
    agent = PPO(WINDOW, env.N, PPO_CFG)
    if os.path.exists(MODEL_PATH): agent.net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    while not stop.is_set():
        time.sleep(5)
        if is_market_open(now_utc()) or not is_nightly(now_utc()):
            continue
        with STATE_LOCK: STATE['status']='training'
        for _ in range(TRAIN_ROUNDS_PER_NIGHT):
            batch=rollout(env,agent,PPO_CFG['rollout_steps']); agent.update(batch)
        torch.save(agent.net.state_dict(), MODEL_PATH)
        with STATE_LOCK: STATE['status']='idle'

# ------------------ LIVE TRADER (paper) ------------------
def live_thread(stop: threading.Event):
    env, tickers = build_env()
    agent = PPO(WINDOW, env.N, PPO_CFG)
    if os.path.exists(MODEL_PATH): agent.net.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))

    equity = 1000.0
    last_prices=None; buf=[]; today_et=now_utc().astimezone(pytz.timezone(MARKET_TZ)).date(); today_pnl=0.0

    with STATE_LOCK: STATE.update({"equity":equity,"today_pnl":0.0,"today_trades":0})

    def log_trade(**kw): insert_trade(kw)

    while not stop.is_set():
        try:
            if not is_market_open(now_utc()):
                with STATE_LOCK: STATE['status']='idle'
                time.sleep(5); continue
            with STATE_LOCK: STATE['status']='live'

            data = yf.download(tickers, period="2d", interval="1m", progress=False, auto_adjust=True)
            closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
            latest = closes.iloc[-1].dropna()
            if last_prices is None:
                last_prices = latest; time.sleep(POLL_SECONDS); continue
            common = latest.index.intersection(last_prices.index)
            rets = (latest[common]/last_prices[common]-1.0)
            last_prices = latest
            buf.append(rets.reindex(tickers).fillna(0.0).values)
            if len(buf)>WINDOW: buf.pop(0)
            if len(buf)<WINDOW:
                time.sleep(POLL_SECONDS); continue

            obs_np = np.stack(buf, axis=0).astype(np.float32)
            with torch.no_grad(): logits,_=agent.net(torch.tensor(obs_np,dtype=torch.float32).unsqueeze(0)); a=int(torch.argmax(logits,dim=-1).item())
            risk = 0.15
            pnl=0.0; act_ticker='CASH'; qty=0.0; price=float(latest.mean()) if len(latest)>0 else 0.0; side='HOLD'
            if a != env.cash:
                r=float(obs_np[-1][a]); act_ticker=tickers[a]
                notional = equity*risk
                qty = notional / max(price,1e-6)
                fee = notional*FEE_BPS
                pnl = notional*(r) - fee
                equity += pnl
                side = 'BUY' if r>=0 else 'SELL'
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
            if today_et_now != today_et:
                today_et = today_et_now
                today_pnl = 0.0
            with STATE_LOCK:
                STATE['today_trades'] = 0

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
def metrics(_: None = Depends(require_key)):
    with STATE_LOCK:
        return JSONResponse(STATE.copy())

@app.get("/trades")
def trades(limit: int = 100, cursor: int = 0, _: None = Depends(require_key)):
    limit = max(1, min(500, limit))
    cur = DB.execute(
        "SELECT * FROM trades WHERE trade_id>? ORDER BY trade_id DESC LIMIT ?",
        (cursor, limit),
    )
    rows = [dict(zip([c[0] for c in cur.description], r)) for r in cur.fetchall()]
    next_cursor = rows[-1]["trade_id"] if rows else cursor
    return {"data": rows, "next_cursor": next_cursor}

@app.get("/equity")
def equity(window: str = "30d", _: None = Depends(require_key)):
    cur = DB.execute("SELECT ts_utc, equity_after FROM trades ORDER BY trade_id ASC")
    rows = cur.fetchall()
    series = [{"t": r[0], "equity": r[1]} for r in rows]
    return {"series": series[-2000:]}

# ------------------ BOOT ------------------
STOP = threading.Event()
threading.Thread(target=trainer_thread, args=(STOP,), daemon=True).start()
threading.Thread(target=live_thread, args=(STOP,), daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
