/* ===== Backend host selection ===== */
const HOSTS = [];
// Netlify proxy: /api/xxx -> backend (same-origin, no CORS)
HOSTS.push({ base: location.origin.replace(/\/$/, ""), prefix: "/api" });
// Direct Render backend (no /api prefix)
if (typeof window !== "undefined" && window.API_FALLBACK) {
  HOSTS.push({ base: String(window.API_FALLBACK).replace(/\/$/, ""), prefix: "" });
}

/* Tiny helpers */
const fmtUSD = (n) => {
  const v = Number(n || 0);
  const s = `$${Math.abs(v).toFixed(2)}`;
  return v >= 0 ? s : `-${s}`;
};
const pct = (x) => `${(Number(x || 0) * 100).toFixed(2)}%`;
const r6  = (x) => Number(x || 0).toFixed(6);

/* Waking badge */
let serverSleeping = false;
function setServerSleeping(flag) {
  serverSleeping = !!flag;
  const el = document.getElementById("serverStatus");
  if (!el) return;
  el.style.display = serverSleeping ? "inline-block" : "none";
  el.textContent = "Waking server… (free instance)";
}
(function ensureStatusBadge() {
  if (document.getElementById("serverStatus")) return;
  const badge = document.createElement("div");
  badge.id = "serverStatus";
  badge.style.cssText =
    "position:absolute;right:18px;top:12px;padding:6px 10px;border-radius:8px;background:#3b3b3b;color:#ffd37a;font-size:12px;display:none;z-index:50;";
  document.body.appendChild(badge);
})();

/* Wake the Render server without tripping CORS (opaque request) */
async function wakeServer() {
  if (!window.API_FALLBACK) return;
  try {
    const base = String(window.API_FALLBACK).replace(/\/$/, "");
    // a couple of pings with increasing gaps
    for (let i = 0; i < 3; i++) {
      try { await fetch(base + "/healthz", { mode: "no-cors", cache: "no-store" }); } catch {}
      await new Promise(r => setTimeout(r, 1200 * (i + 1)));
    }
  } catch {}
}
wakeServer();

/* Robust fetch with fallback + timeout */
async function fetchJSONWithFallback(path, { method = "GET", body, timeoutMs = 15000 } = {}) {
  setServerSleeping(true);
  let lastErr;
  for (const host of HOSTS) {
    const url = host.base + host.prefix + path;
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), timeoutMs);
      const res = await fetch(url, {
        method,
        cache: "no-store",
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: ctrl.signal
      });
      clearTimeout(t);
      if (res.ok) {
        setServerSleeping(false);
        return res.json();
      }
      lastErr = new Error(`${res.status}`);
      // try next host
    } catch (e) {
      lastErr = e;
      // try next host
    }
  }
  setServerSleeping(true);
  throw lastErr || new Error("unreachable");
}

const getJSON  = (p) => fetchJSONWithFallback(p);
const postJSON = (p, body) => fetchJSONWithFallback(p, { method: "POST", body });

/* Adaptive poller with backoff */
function makePoller(fn, { min = 5000, max = 45000, step = 1.8 } = {}) {
  let delay = min, timer = null, running = false;

  async function run() {
    if (running) return;
    running = true;
    try {
      await fn();
      delay = min;
      setServerSleeping(false);
    } catch (e) {
      delay = Math.min(max, Math.ceil(delay * step));
      setServerSleeping(true);
    } finally {
      running = false;
      timer = setTimeout(run, delay);
    }
  }
  run();
  return { stop: () => timer && clearTimeout(timer) };
}

/* Admin: Reset */
document.getElementById("resetBtn")?.addEventListener("click", async () => {
  const pwd = prompt("Admin password to reset all model & trades:");
  if (!pwd) return;
  try {
    await fetchJSONWithFallback("/admin/reset", { method: "POST", body: { password: pwd } });
    alert("Reset complete.");
  } catch (e) {
    alert(`Reset failed: ${e.message}`);
  }
});

/* Benchmark control (click the pill) */
const BENCH_KEY = "benchSymbol";
const benchSymbol = () => localStorage.getItem(BENCH_KEY) || "SPY";
const setBench = (s) => localStorage.setItem(BENCH_KEY, (s || "SPY").toUpperCase());
document.getElementById("benchLabel")?.addEventListener("click", async () => {
  const sym = prompt("Benchmark ticker (e.g., SPY, QQQ):", benchSymbol());
  if (!sym) return;
  setBench(sym);
  document.getElementById("benchLabel").textContent = benchSymbol();
  try { await loadEquity(); } catch {}
});

/* Metrics */
async function loadMetrics() {
  const m = await fetchJSONWithFallback("/metrics");
  document.getElementById("mode").innerHTML =
    `<b>${(m.mode || m.status || "idle").toString().toUpperCase()}</b>` +
    (String(m.status || "").startsWith("train-error") ? " ⚠️" : "");
  document.getElementById("equity").textContent = fmtUSD(m.equity);
  document.getElementById("pnl").textContent = fmtUSD(m.today_pnl);
  document.getElementById("trades").textContent = Number(m.today_trades || 0);
  document.getElementById("universe").textContent = Number(m.universe || 0);
  document.getElementById("dd").textContent = pct(m.max_drawdown);
  document.getElementById("blockRisk").textContent =
    m.block_risk != null ? Number(m.block_risk).toFixed(3) : "–";
  document.getElementById("blockCash").textContent =
    m.block_cash != null ? fmtUSD(m.block_cash) : "–";
  document.getElementById("cash").textContent = fmtUSD(m.cash);
  document.getElementById("invested").textContent = fmtUSD(m.positions_value);
  document.getElementById("upnl").textContent = fmtUSD(m.unrealized_pnl);

  const train = m.train || {};
  document.getElementById("rewardMean").textContent =
    Number(train.last_reward_mean || 0).toFixed(4);
  document.getElementById("winRate").textContent =
    `${(Number(train.last_win_rate || 0) * 100).toFixed(1)}%`;
  document.getElementById("lastTrain").textContent =
    train.last_time_utc ? new Date(train.last_time_utc).toLocaleString() : "–";
}

/* Equity chart + benchmark */
let eqChart;
async function loadEquity() {
  const bench = benchSymbol();
  const payload = await fetchJSONWithFallback(`/equity?bench=${encodeURIComponent(bench)}`, { timeoutMs: 12000 });
  const { series = [], bench: benchSeries = [] } = payload;

  const equityPts = series.map(p => ({ x: p.t, y: Number(p.equity) }));
  const benchPts  = benchSeries.map(p => ({ x: p.t, y: Number(p.equity) }));

  const bEl = document.getElementById("benchLabel");
  if (bEl) bEl.textContent = bench;

  const ctx = document.getElementById("equityChart").getContext("2d");
  if (!eqChart) {
    eqChart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          {
            label: "Equity",
            data: equityPts,
            tension: 0.25,
            pointRadius: 0,
            borderWidth: 2,
            borderColor: "#12d1c1",
            backgroundColor: "rgba(18, 209, 193, 0.15)"
          },
          {
            label: `Benchmark (${bench})`,
            data: benchPts,
            tension: 0.25,
            pointRadius: 0,
            borderWidth: 2,
            borderDash: [6, 4],
            borderColor: "#ff6b6b"
          }
        ]
      },
      options: {
        responsive: true,
        animation: false,
        parsing: true,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "time", time: { tooltipFormat: "MMM d, HH:mm" },
               grid: { color: "rgba(255,255,255,0.06)" }, ticks: { color: "rgba(255,255,255,0.7)" } },
          y: { beginAtZero: false,
               grid: { color: "rgba(255,255,255,0.06)" }, ticks: { color: "rgba(255,255,255,0.7)" } }
        },
        plugins: { legend: { display: true, labels: { color: "rgba(255,255,255,0.8)" } } }
      }
    });
  } else {
    eqChart.data.datasets[0].data = equityPts;
    eqChart.data.datasets[1].data = benchPts;
    eqChart.data.datasets[1].label = `Benchmark (${bench})`;
    eqChart.update();
  }
}

/* Trades */
let tradesCursor = 0;
async function loadTrades() {
  const payload = await fetchJSONWithFallback(`/trades?limit=100&cursor=${tradesCursor}`);
  tradesCursor = payload.next_cursor || tradesCursor;
  const rows = payload.data || [];
  const tbody = document.getElementById("tradesBody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const side = (r.side || "").toUpperCase();
    const pnl = Number(r.realized_pnl || 0);
    const showPnl = side === "SELL";
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.trade_id}</td>
      <td>${r.mode}</td>
      <td>${new Date(r.ts_et || r.ts_utc).toLocaleString()}</td>
      <td>${side}</td>
      <td>${r.ticker}</td>
      <td>${r6(r.qty)}</td>
      <td>${Number(r.fill_price||0).toFixed(2)}</td>
      <td>${Number(r.notional||0).toFixed(2)}</td>
      <td>${Number(r.risk_frac||0).toFixed(3)}</td>
      <td class="${showPnl ? (pnl>=0?'good':'bad') : ''}">${showPnl ? pnl.toFixed(2) : "–"}</td>
      <td>${Number(r.equity_after||0).toFixed(2)}</td>
      <td>${r.reason||""}</td>`;
    tbody.appendChild(tr);
  }
}

/* Secondary stats */
async function loadStats() {
  const s = await fetchJSONWithFallback("/stats");
  const train = s.train || {};
  document.getElementById("rewardMean").textContent = Number(train.last_reward_mean || 0).toFixed(4);
  document.getElementById("winRate").textContent = `${(Number(train.last_win_rate || 0) * 100).toFixed(1)}%`;
  document.getElementById("lastTrain").textContent =
    train.last_time_utc ? new Date(train.last_time_utc).toLocaleString() : "–";
}

/* Start adaptive pollers */
makePoller(loadMetrics, { min: 4000, max: 30000, step: 1.8 });
makePoller(loadTrades,  { min: 7000, max: 45000, step: 1.8 });
makePoller(loadEquity,  { min: 12000, max: 60000, step: 1.8 });
makePoller(loadStats,   { min: 20000, max: 60000, step: 1.8 });

/* Initial badge state hidden */
setServerSleeping(false);
