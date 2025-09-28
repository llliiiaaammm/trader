const API_BASE = "/api";

/* ---------- helpers ---------- */
const fmtUSD = (n) => {
  const v = Number(n || 0);
  const s = `$${Math.abs(v).toFixed(2)}`;
  return v >= 0 ? s : `-${s}`;
};
const pct = (x) => `${(Number(x || 0) * 100).toFixed(2)}%`;

/* fetch with a short timeout so we don’t stall the UI */
async function getJSON(path, { timeoutMs = 4000 } = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, { cache: "no-store", signal: ctrl.signal });
    if (!res.ok) throw new Error(String(res.status));
    return await res.json();
  } finally { clearTimeout(t); }
}
async function postJSON(path, body, { timeoutMs = 8000 } = {}) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
      signal: ctrl.signal
    });
    if (!res.ok) throw new Error(String(res.status));
    return await res.json();
  } finally { clearTimeout(t); }
}

/* ---------- hide controls (functionality still on server) ---------- */
for (const id of ["pauseBtn","resumeBtn","tradeBtn"]) {
  document.getElementById(id)?.remove();
}

/* ---------- metrics panel ---------- */
async function loadMetrics() {
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
  } catch {
    // leave the last values in place; try again next tick
  }
}

/* ---------- training stats ---------- */
async function loadStats() {
  try {
    const s = await getJSON("/stats");
    const state = s.state || {}, train = s.train || {};
    const reward = state.train_reward_mean ?? train.last_reward_mean ?? 0;
    const win    = state.train_win_rate   ?? train.last_win_rate   ?? 0;
    document.getElementById("rewardMean").textContent = Number(reward).toFixed(4);
    document.getElementById("winRate").textContent = `${(Number(win) * 100).toFixed(1)}%`;
    document.getElementById("lastTrain").textContent =
      (state.last_train_utc || train.last_time_utc) ? new Date(state.last_train_utc || train.last_time_utc).toLocaleString() : "–";
  } catch {}
}

/* ---------- equity chart (lazy benchmark) ---------- */
let eqChart;
function getBenchTicker() {
  return (localStorage.getItem("benchTicker") || "SPY").toUpperCase();
}
window.setBenchmarkTicker = function (t) {
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
  try {
    // fast first: no benchmark so we paint immediately
    const fast = await getJSON("/equity?bench=0");
    const equityPts = (fast.series || []).map(p => ({ x: p.t, y: Number(p.equity) }));
    const ctx = document.getElementById("equityChart").getContext("2d");
    if (!eqChart) {
      eqChart = new Chart(ctx, {
        type: "line",
        data: { datasets: [
          { label: "Equity", data: equityPts, tension: 0.25, pointRadius: 0, borderWidth: 2, fill: false },
          { label: `Benchmark (${getBenchTicker()})`, data: [], tension: 0.25, pointRadius: 0, borderWidth: 2 }
        ]},
        options: {
          responsive: true, animation: false, parsing: true,
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
    }
    eqChart.update();

    // then lazy benchmark; if it times out, we’ll try again next refresh
    const tkr = encodeURIComponent(getBenchTicker());
    getJSON(`/equity?bench=1&bench_ticker=${tkr}`).then(full => {
      const benchPts = (full.bench || []).map(p => ({ x: p.t, y: Number(p.equity) }));
      eqChart.data.datasets[1].label = `Benchmark (${getBenchTicker()})`;
      if (benchPts.length) eqChart.data.datasets[1].data = benchPts;
      eqChart.update();
    }).catch(() => {});
  } catch {
    // nothing to do; next refresh will try again
  }
}

/* ---------- trades ---------- */
let tradesCursor = 0;
async function loadTrades() {
  try {
    const payload = await getJSON(`/trades?limit=100&cursor=${tradesCursor}`);
    tradesCursor = payload.next_cursor || tradesCursor;
    const rows = payload.data || [];
    const tbody = document.getElementById("tradesBody");
    tbody.innerHTML = "";
    for (const r of rows) {
      const tr = document.createElement("tr");
      const side = (r.side === "SIM" ? "BUY" : r.side);
      const pnl = Number(r.realized_pnl || 0);
      const pnlDisplay = side === "SELL" ? pnl.toFixed(2) : "";
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
  } catch {
    // keep the previous table; try again next tick
  }
}

/* ---------- refresh loop (never blocked by /healthz) ---------- */
async function refresh() {
  try {
    await Promise.all([loadMetrics(), loadEquity(), loadTrades(), loadStats()]);
  } catch {}
  setTimeout(refresh, 5000);
}
refresh();
