const API_BASE = "/api";

// Hide removed controls (functionality still exists server-side)
["pauseBtn", "resumeBtn", "manualBtn"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.style.display = "none";
});

// ---------- helpers ----------
const fmtUSD = (n) => {
  const v = Number(n || 0);
  const s = `$${Math.abs(v).toFixed(2)}`;
  return v >= 0 ? s : `-${s}`;
};
const pct = (x) => `${(Number(x || 0) * 100).toFixed(2)}%`;
const r6 = (x) => Number(x || 0).toFixed(6);

// small fetch with retry/backoff to cope with free-instance cold starts
async function getJSON(path, { retries = 2 } = {}) {
  const url = `${API_BASE}${path}`;
  let last;
  for (let i = 0; i <= retries; i++) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (r.ok) return r.json();
      last = r.status;
    } catch (e) {
      last = e.message;
    }
    await new Promise((r) => setTimeout(r, 500 * (i + 1)));
  }
  throw new Error(String(last || "fetch failed"));
}

async function postJSON(path, body) {
  const r = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

// ---------- wake gate (prevents spamming 503s) ----------
async function waitForReady() {
  const t0 = Date.now();
  while (Date.now() - t0 < 60000) { // 60s max
    try {
      const h = await getJSON("/healthz", { retries: 0 });
      if (h && h.ok) return true;
    } catch {}
    await new Promise((r) => setTimeout(r, 800));
  }
  return false;
}

// ---------- admin: Reset (only one left) ----------
document.getElementById("resetBtn")?.addEventListener("click", async () => {
  const pwd = prompt("Admin password to reset all state:");
  if (!pwd) return;
  try { await postJSON("/admin/reset", { password: pwd }); alert("Reset complete"); }
  catch (e) { alert(`Reset failed: ${e.message}`); }
});

// ---------- benchmark control ----------
const BENCH_KEY = "benchSymbol";
const benchSymbol = () => localStorage.getItem(BENCH_KEY) || "SPY";
const setBench = (s) => localStorage.setItem(BENCH_KEY, (s || "SPY").toUpperCase());
document.getElementById("benchLabel")?.addEventListener("click", async () => {
  const sym = prompt("Benchmark ticker (e.g., SPY, QQQ):", benchSymbol());
  if (!sym) return;
  setBench(sym);
  document.getElementById("benchLabel").textContent = benchSymbol();
  await loadEquity(true);
});

// ---------- metrics ----------
async function loadMetrics() {
  const m = await getJSON("/metrics");
  document.getElementById("mode").innerHTML = `<b>${m.status || "idle"}</b>`;
  document.getElementById("equity").textContent = fmtUSD(m.equity);
  document.getElementById("pnl").textContent = fmtUSD(m.today_pnl);
  document.getElementById("trades").textContent = Number(m.today_trades || 0);
  document.getElementById("universe").textContent = Number(m.universe || 0);
  document.getElementById("dd").textContent = pct(m.max_drawdown);
  document.getElementById("blockRisk").textContent = m.block_risk != null ? (Number(m.block_risk).toFixed(3)) : "–";
  document.getElementById("blockCash").textContent = m.block_cash != null ? fmtUSD(m.block_cash) : "–";
  document.getElementById("cash").textContent = fmtUSD(m.cash);
  document.getElementById("invested").textContent = fmtUSD(m.positions_value);
  document.getElementById("upnl").textContent = fmtUSD(m.unrealized_pnl);
}

// ---------- equity chart ----------
let eqChart;
async function loadEquity() {
  const bench = benchSymbol();
  const payload = await getJSON(`/equity?bench=${encodeURIComponent(bench)}`);
  const { series = [], bench: benchSeries = [] } = payload;

  const equityPts = series.map(p => ({ x: p.t, y: Number(p.equity) }));
  const benchPts  = benchSeries.map(p => ({ x: p.t, y: Number(p.equity) }));

  const bEl = document.getElementById("benchLabel");
  if (bEl) bEl.textContent = bench;

  const ctx = document.getElementById("equityChart").getContext("2d");
  if (!eqChart) {
    eqChart = new Chart(ctx, {
      type: "line",
      data: { datasets: [
        { label: "Equity", data: equityPts, tension: 0.25, pointRadius: 0, borderWidth: 2 },
        { label: `Benchmark (${bench})`, data: benchPts, tension: 0.25, pointRadius: 0, borderWidth: 2, borderColor: "#ff6b6b" }
      ]},
      options: {
        responsive: true, animation: false, parsing: true,
        interaction: { mode: "nearest", intersect: false },
        scales: { x: { type: "time", time: { tooltipFormat: "MMM d, HH:mm" } }, y: { beginAtZero: false } },
        plugins: { legend: { display: true } }
      }
    });
  } else {
    eqChart.data.datasets[0].data = equityPts;
    eqChart.data.datasets[1].data = benchPts;
    eqChart.data.datasets[1].label = `Benchmark (${bench})`;
    eqChart.update();
  }
}

// ---------- trades ----------
let tradesCursor = 0;
async function loadTrades() {
  const payload = await getJSON(`/trades?limit=100&cursor=${tradesCursor}`);
  tradesCursor = payload.next_cursor || tradesCursor;
  const rows = payload.data || [];
  const tbody = document.getElementById("tradesBody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const pnl = Number(r.realized_pnl || 0);
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
      <td class="${pnl>=0?'good':'bad'}">${pnl.toFixed(2)}</td>
      <td>${Number(r.equity_after||0).toFixed(2)}</td>
      <td>${r.reason||""}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ---------- training stats ----------
async function loadStats() {
  const s = await getJSON("/stats");
  const train = s.train || {};
  document.getElementById("rewardMean").textContent = Number(train.last_reward_mean || 0).toFixed(4);
  document.getElementById("winRate").textContent = `${(Number(train.last_win_rate || 0) * 100).toFixed(1)}%`;
  document.getElementById("lastTrain").textContent =
    train.last_time_utc ? new Date(train.last_time_utc).toLocaleString() : "–";
}

// ---------- polling (staggered) ----------
async function boot() {
  // wait for server to wake before starting the loops
  await waitForReady();

  // initial burst
  try { await loadMetrics(); } catch {}
  try { await loadEquity(); }  catch {}
  try { await loadTrades(); }  catch {}
  try { await loadStats(); }   catch {}

  // recurring
  setInterval(() => loadMetrics().catch(() => {}),  5000);
  setInterval(() => loadTrades().catch(() => {}),  10000);
  setInterval(() => loadEquity().catch(() => {}), 30000);
  setInterval(() => loadStats().catch(() => {}),  30000);
}
boot();
