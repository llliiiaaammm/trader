// app.js
const API_BASE = "/api";

// ---- helpers
const fmtUSD = (n) => {
  const v = Number(n || 0);
  const s = `$${Math.abs(v).toFixed(2)}`;
  return v >= 0 ? s : `-${s}`;
};
const pct = (x) => `${(Number(x || 0) * 100).toFixed(2)}%`;
const r6 = (x) => Number(x || 0).toFixed(6);

// ---- generic fetch with small retry (helps during cold-start)
async function getJSON(path) {
  const url = `${API_BASE}${path}`;
  for (let i = 0; i < 2; i++) {
    const res = await fetch(url, { cache: "no-store" }).catch(() => null);
    if (res && res.ok) return res.json();
    await new Promise((r) => setTimeout(r, 400));
  }
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

async function postJSON(path, body) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

// ---- Admin: only Reset is visible in UI. Pause/Resume/Manual trade remain in backend.
document.getElementById("resetBtn")?.addEventListener("click", async () => {
  const pwd = prompt("Admin password to reset all model & trades:");
  if (!pwd) return;
  try {
    await postJSON("/admin/reset", { password: pwd });
    alert("Reset complete. Service will reinitialize.");
  } catch (e) {
    alert(`Reset failed: ${e.message}`);
  }
});

// ---- Benchmark control (click the label to change)
const BENCH_KEY = "benchSymbol";
function benchSymbol() { return localStorage.getItem(BENCH_KEY) || "SPY"; }
function setBench(sym) { localStorage.setItem(BENCH_KEY, (sym || "SPY").toUpperCase()); }
document.getElementById("benchLabel")?.addEventListener("click", async () => {
  const cur = benchSymbol();
  const sym = prompt("Benchmark ticker (e.g., SPY, QQQ):", cur);
  if (!sym) return;
  setBench(sym.toUpperCase());
  document.getElementById("benchLabel").textContent = benchSymbol();
  // force chart refresh now
  await loadEquity(true);
});

// ---- metrics
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

// ---- equity chart with benchmark
let eqChart;
async function loadEquity(force = false) {
  const bench = benchSymbol();
  const payload = await getJSON(`/equity?bench=${encodeURIComponent(bench)}`);
  const { series = [], bench: benchSeries = [] } = payload;
  const equityPts = series.map(p => ({ x: p.t, y: Number(p.equity) }));
  const benchPts  = benchSeries.map(p => ({ x: p.t, y: Number(p.equity) }));

  // show current benchmark symbol in the side panel
  const bEl = document.getElementById("benchLabel");
  if (bEl) bEl.textContent = bench;

  if (!eqChart) {
    const ctx = document.getElementById("equityChart").getContext("2d");
    eqChart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          { label: "Equity", data: equityPts, tension: 0.25, pointRadius: 0 },
          { label: `Benchmark (${bench})`, data: benchPts, tension: 0.25, pointRadius: 0, borderColor: "#ff6b6b" }
        ]
      },
      options: {
        responsive: true,
        animation: false,
        parsing: true,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "time", time: { tooltipFormat: "MMM d, HH:mm" } },
          y: { beginAtZero: false }
        },
        plugins: { legend: { display: true } }
      }
    });
  } else {
    eqChart.data.datasets[0].label = "Equity";
    eqChart.data.datasets[0].data = equityPts;
    eqChart.data.datasets[1].label = `Benchmark (${bench})`;
    eqChart.data.datasets[1].data = benchPts;
    eqChart.update();
  }
}

// ---- trades table
let tradesCursor = 0;
async function loadTrades() {
  const payload = await getJSON(`/trades?limit=100&cursor=${tradesCursor}`);
  tradesCursor = payload.next_cursor || tradesCursor;
  const rows = payload.data || [];
  const tbody = document.getElementById("tradesBody");
  tbody.innerHTML = "";
  for (const r of rows) {
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
      <td class="${pnl>=0?'good':'bad'}">${pnl.toFixed(2)}</td>
      <td>${Number(r.equity_after||0).toFixed(2)}</td>
      <td>${r.reason||""}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ---- training stats
async function loadStats() {
  const s = await getJSON("/stats");
  const state = s.state || {}, train = s.train || {};
  const reward = train.last_reward_mean ?? 0;
  const win    = train.last_win_rate   ?? 0;

  document.getElementById("rewardMean").textContent = Number(reward).toFixed(4);
  document.getElementById("winRate").textContent = `${(Number(win) * 100).toFixed(1)}%`;
  document.getElementById("lastTrain").textContent =
    train.last_time_utc ? new Date(train.last_time_utc).toLocaleString() : "–";
}

// ---- refresh loop
async function refresh() {
  try {
    await Promise.all([loadMetrics(), loadEquity(), loadTrades(), loadStats()]);
  } catch (e) {
    console.warn("UI refresh error:", e.message);
  } finally {
    setTimeout(refresh, 5000);
  }
}
refresh();
