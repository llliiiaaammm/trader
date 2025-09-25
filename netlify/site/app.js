const API_BASE = "/api";

// ===== Helpers =====
const fmtUSD = (n) => {
  const v = Number(n || 0);
  const s = `$${Math.abs(v).toFixed(2)}`;
  return v >= 0 ? s : `-${s}`;
};
const pct = (x) => `${(Number(x || 0) * 100).toFixed(2)}%`;

// small fetch wrappers
async function getJSON(path) {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}
async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

// ===== Remove controls in UI (functionality still exists server-side) =====
for (const id of ["pauseBtn","resumeBtn","tradeBtn"]) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ===== Health warmup =====
async function waitForApiReady() {
  for (let i = 0; i < 10; i++) {
    try { const r = await fetch(`${API_BASE}/healthz`); if (r.ok) return; } catch {}
    await new Promise(r => setTimeout(r, 800));
  }
}

// ===== Metrics panel =====
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

// ===== Training stats left panel =====
async function loadStats() {
  const s = await getJSON("/stats");
  const state = s.state || {}, train = s.train || {};
  const reward = state.train_reward_mean ?? train.last_reward_mean ?? 0;
  const win    = state.train_win_rate   ?? train.last_win_rate   ?? 0;

  document.getElementById("rewardMean").textContent = Number(reward).toFixed(4);
  document.getElementById("winRate").textContent = `${(Number(win) * 100).toFixed(1)}%`;
  document.getElementById("lastTrain").textContent =
    (state.last_train_utc || train.last_time_utc) ? new Date(state.last_train_utc || train.last_time_utc).toLocaleString() : "–";
}

// ===== Equity chart (benchmark selectable) =====
let eqChart;
function getBenchTicker() {
  return (localStorage.getItem("benchTicker") || "SPY").toUpperCase();
}
window.setBenchmarkTicker = function (t) {  // quick way to change via console: setBenchmarkTicker('QQQ')
  const clean = (t || "").trim().toUpperCase();
  if (clean) localStorage.setItem("benchTicker", clean);
};
window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "b") {
    const cur = getBenchTicker();
    const t = prompt("Benchmark ticker:", cur);
    if (t) localStorage.setItem("benchTicker", t.trim().toUpperCase());
  }
});

async function loadEquity() {
  const benchTicker = encodeURIComponent(getBenchTicker());
  const { series = [], bench = [] } = await getJSON(`/equity?bench=1&bench_ticker=${benchTicker}`);

  const equityPts = series.map(p => ({ x: p.t, y: Number(p.equity) }));
  const benchPts  = (bench || []).map(p => ({ x: p.t, y: Number(p.equity) }));

  const ctx = document.getElementById("equityChart").getContext("2d");
  if (!eqChart) {
    eqChart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [
          { label: "Equity", data: equityPts, tension: 0.25, pointRadius: 0, borderWidth: 2, fill: false },
          { label: `Benchmark (${getBenchTicker()})`, data: benchPts, tension: 0.25, pointRadius: 0, borderWidth: 2 }
        ]
      },
      options: {
        responsive: true,
        animation: false,
        parsing: true,
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: { type: "time", time: { tooltipFormat: "MMM d, HH:mm" }, ticks: { color: "#a0a0a0" }, grid: { color: "rgba(255,255,255,0.06)" } },
          y: { beginAtZero: false, ticks: { color: "#a0a0a0" }, grid: { color: "rgba(255,255,255,0.06)" } }
        },
        plugins: { legend: { display: true } }
      }
    });
  } else {
    eqChart.data.datasets[0].data = equityPts;
    eqChart.data.datasets[1].label = `Benchmark (${getBenchTicker()})`;
    if (benchPts.length) eqChart.data.datasets[1].data = benchPts;
    eqChart.update();
  }
}

// ===== Trades table =====
let tradesCursor = 0;
async function loadTrades() {
  const payload = await getJSON(`/trades?limit=100&cursor=${tradesCursor}`);
  tradesCursor = payload.next_cursor || tradesCursor;
  const rows = payload.data || [];
  const tbody = document.getElementById("tradesBody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    const side = (r.side === "SIM" ? "BUY" : r.side); // safety if any SIM slipped through
    const pnl = Number(r.realized_pnl || 0);
    const pnlDisplay = side === "SELL" ? pnl.toFixed(2) : ""; // only show PnL for sells
    tr.innerHTML = `
      <td>${r.trade_id}</td>
      <td>${r.mode}</td>
      <td>${new Date(r.ts_et || r.ts_utc).toLocaleString()}</td>
      <td>${side}</td>
      <td>${r.ticker}</td>
      <td>${Number(r.qty||0).toFixed(6)}</td>
      <td>${Number(r.fill_price||0).toFixed(2)}</td>
      <td>${Number(r.notional||0).toFixed(2)}</td>
      <td>${Number(r.risk_frac||0).toFixed(3)}</td>
      <td class="${pnl>=0?'good':'bad'}">${pnlDisplay}</td>
      <td>${Number(r.equity_after||0).toFixed(2)}</td>
      <td>${r.reason||""}</td>
    `;
    tbody.appendChild(tr);
  }
}

// ===== Refresh loop =====
async function refresh() {
  try {
    await Promise.all([loadMetrics(), loadEquity(), loadTrades(), loadStats()]);
  } catch (e) {
    console.warn("UI refresh error:", e.message);
  } finally {
    setTimeout(refresh, 5000);
  }
}

(async () => { await waitForApiReady(); refresh(); })();
