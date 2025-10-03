/* -------- API base detection (root or /api) -------- */
let API_BASE = null;
async function ensureBase() {
  if (API_BASE !== null) return API_BASE;
  const tryBases = ["/api", ""];
  for (const base of tryBases) {
    try { const r = await fetch(`${base}/healthz`, { cache: "no-store" }); if (r.ok) { API_BASE = base; break; } } catch {}
  }
  API_BASE = API_BASE ?? (localStorage.getItem("apiBase") || "/api");
  localStorage.setItem("apiBase", API_BASE);
  return API_BASE;
}
async function getJSON(path) {
  const base = await ensureBase();
  const url = `${base}${path}`;
  try { const r = await fetch(url, { cache: "no-store" }); if (r.ok) return r.json(); } catch {}
  const r2 = await fetch(url, { cache: "no-store" });
  if (!r2.ok) throw new Error(`${r2.status} on ${url}`);
  return r2.json();
}
async function postJSON(path, body) {
  const base = await ensureBase();
  const r = await fetch(`${base}${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

/* -------- Hide Pause/Resume/Manual trade everywhere -------- */
["pauseBtn", "resumeBtn", "manualBtn", "tradeBtn", "tradeBox"].forEach((id) => {
  const el = document.getElementById(id); if (el) el.style.display = "none";
});
Array.from(document.querySelectorAll("button")).forEach(btn => {
  if (/pause|resume|manual/i.test(btn.textContent || "")) btn.style.display = "none";
});

/* -------- Helpers -------- */
const fmtUSD = (n) => { const v = Number(n || 0); const s = `$${Math.abs(v).toFixed(2)}`; return v >= 0 ? s : `-${s}`; };
const pct = (x) => `${(Number(x || 0) * 100).toFixed(2)}%`;
const r6 = (x) => Number(x || 0).toFixed(6);

/* -------- Benchmark picker -------- */
const BENCH_KEY = "benchSymbol";
const benchSymbol = () => localStorage.getItem(BENCH_KEY) || "SPY";
const setBench = (s) => localStorage.setItem(BENCH_KEY, (s || "SPY").toUpperCase());
document.getElementById("benchLabel")?.addEventListener("click", async () => {
  const sym = prompt("Benchmark ticker (e.g., SPY, QQQ):", benchSymbol());
  if (!sym) return;
  setBench(sym);
  document.getElementById("benchLabel").textContent = benchSymbol();
  await loadEquity();
});

/* -------- Sections (non-overlapping polling) -------- */
const inflight = { metrics:false, trades:false, equity:false, stats:false };

async function loadMetrics() {
  if (inflight.metrics) return; inflight.metrics = true;
  try {
    const m = await getJSON("/metrics");
    document.getElementById("mode").innerHTML = `<b>${m.status || "idle"}</b>`;
    document.getElementById("equity").textContent = fmtUSD(m.equity);
    document.getElementById("pnl").textContent = fmtUSD(m.today_pnl);
    document.getElementById("trades").textContent = Number(m.today_trades || 0);
    document.getElementById("universe").textContent = Number(m.universe || 0);
    document.getElementById("dd").textContent = pct(m.max_drawdown);
    document.getElementById("blockRisk").textContent = m.block_risk != null ? Number(m.block_risk).toFixed(3) : "–";
    document.getElementById("blockCash").textContent = m.block_cash != null ? fmtUSD(m.block_cash) : "–";
    document.getElementById("cash").textContent = fmtUSD(m.cash);
    document.getElementById("invested").textContent = fmtUSD(m.positions_value);
    document.getElementById("upnl").textContent = fmtUSD(m.unrealized_pnl);
  } finally { inflight.metrics = false; }
}

let eqChart;
async function loadEquity() {
  if (inflight.equity) return; inflight.equity = true;
  try {
    const bench = benchSymbol();
    const payload = await getJSON(`/equity?bench=${encodeURIComponent(bench)}`);
    const { series = [], bench: benchSeries = [] } = payload;
    const equityPts = series.map(p => ({ x: p.t, y: Number(p.equity) }));
    const benchPts  = benchSeries.map(p => ({ x: p.t, y: Number(p.equity) }));
    const labelEl = document.getElementById("benchLabel"); if (labelEl) labelEl.textContent = bench;
    const ctx = document.getElementById("equityChart")?.getContext?.("2d"); if (!ctx || !window.Chart) return;
    if (!eqChart) {
      eqChart = new Chart(ctx, {
        type: "line",
        data: {
          datasets: [
            { label: "Equity", data: equityPts, tension: 0.25, pointRadius: 0 },
            { label: `Benchmark (${bench})`, data: benchPts, tension: 0.25, pointRadius: 0, borderColor: "#ff6b6b" }
          ]
        },
        options: {
          responsive: true, animation: false, parsing: true,
          interaction: { mode: "nearest", intersect: false },
          scales: { x: { type: "time" }, y: { beginAtZero: false } },
          plugins: { legend: { display: true } }
        }
      });
    } else {
      eqChart.data.datasets[0].data = equityPts;
      eqChart.data.datasets[1].data = benchPts;
      eqChart.data.datasets[1].label = `Benchmark (${bench})`;
      eqChart.update();
    }
  } finally { inflight.equity = false; }
}

let tradesCursor = 0;
async function loadTrades() {
  if (inflight.trades) return; inflight.trades = true;
  try {
    const payload = await getJSON(`/trades?limit=100&cursor=${tradesCursor}`);
    tradesCursor = payload.next_cursor || tradesCursor;
    const rows = payload.data || [];
    const tbody = document.getElementById("tradesBody"); if (!tbody) return;
    tbody.innerHTML = "";
    for (const r of rows) {
      const showPnl = (r.side || "").toUpperCase() === "SELL";
      const pnl = Number(r.realized_pnl||0);
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${r.trade_id}</td>
        <td>${r.mode}</td>
        <td>${new Date(r.ts_et || r.ts_utc).toLocaleString()}</td>
        <td>${r.side}</td>
        <td>${r.ticker}</td>
        <td>${r6(r.qty)}</td>
        <td>${Number(r.fill_price||0).toFixed(2)}</td>
        <td>${Number(r.notional||0).toFixed(2)}</td>
        <td>${Number(r.risk_frac||0).toFixed(3)}</td>
        <td class="${showPnl ? (pnl>=0?'good':'bad') : ''}">${showPnl ? pnl.toFixed(2) : ""}</td>
        <td>${Number(r.equity_after||0).toFixed(2)}</td>
        <td>${r.reason||""}</td>`;
      tbody.appendChild(tr);
    }
  } finally { inflight.trades = false; }
}

async function loadStats() {
  if (inflight.stats) return; inflight.stats = true;
  try {
    const s = await getJSON("/stats");
    const train = s.train || {};
    document.getElementById("rewardMean").textContent = Number(train.last_reward_mean || 0).toFixed(4);
    document.getElementById("winRate").textContent = `${(Number(train.last_win_rate || 0) * 100).toFixed(1)}%`;
    document.getElementById("lastTrain").textContent = train.last_time_utc ? new Date(train.last_time_utc).toLocaleString() : "–";
  } finally { inflight.stats = false; }
}

/* -------- Polling (staggered, non-overlapping) -------- */
(async () => {
  await ensureBase();
  await loadMetrics(); await loadEquity(); await loadTrades(); await loadStats();
  setInterval(() => loadMetrics().catch(()=>{}), 6000);
  setInterval(() => loadTrades().catch(()=>{}), 12000);
  setInterval(() => loadEquity().catch(()=>{}), 35000);
  setInterval(() => loadStats().catch(()=>{}), 35000);
})();
