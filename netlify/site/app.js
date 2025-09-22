const API_BASE = "/api";

async function getJSON(path) {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function postJSON(path, body) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

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

const fmtUSD = (n) => (Number(n) >= 0 ? `$${Number(n).toFixed(2)}` : `-$${Math.abs(Number(n)).toFixed(2)}`);
const pct = (x) => `${(Number(x) * 100).toFixed(2)}%`;

async function loadMetrics() {
  const m = await getJSON("/metrics");
  // Backend exposes: mode/status/equity/cash/positions_value/unrealized_pnl/today_pnl/today_trades/universe/max_drawdown
  document.getElementById("mode").textContent = m.status || "idle";
  document.getElementById("equity").textContent = fmtUSD(m.equity || 0);
  document.getElementById("pnl").textContent = fmtUSD(m.today_pnl || 0);
  document.getElementById("trades").textContent = Number(m.today_trades || 0);

  // New fields:
  document.getElementById("cash").textContent = fmtUSD(m.cash || 0);
  document.getElementById("invested").textContent = fmtUSD(m.positions_value || 0);
  document.getElementById("upnl").textContent = fmtUSD(m.unrealized_pnl || 0);

  document.getElementById("universe").textContent = Number(m.universe || 0);
  document.getElementById("dd").textContent = pct(m.max_drawdown || 0);
}

let eqChart;
async function loadEquity() {
  const { series = [] } = await getJSON("/equity");
  const points = series.map(p => ({ x: p.t, y: Number(p.equity) }));

  if (!eqChart) {
    const ctx = document.getElementById("equityChart").getContext("2d");
    eqChart = new Chart(ctx, {
      type: "line",
      data: { datasets: [{ label: "Equity", data: points, tension: 0.2, pointRadius: 0 }] },
      options: {
        animation: false,
        parsing: true,
        scales: {
          x: { type: "time", time: { tooltipFormat: "MMM d, HH:mm" } },
          y: { beginAtZero: false }
        },
        plugins: { legend: { display: true } }
      }
    });
  } else {
    eqChart.data.datasets[0].data = points;
    eqChart.update();
  }
}

let tradesCursor = 0;
async function loadTrades() {
  const payload = await getJSON(`/trades?limit=100&cursor=${tradesCursor}`);
  tradesCursor = payload.next_cursor || tradesCursor;
  const rows = payload.data || [];
  const tbody = document.getElementById("tradesBody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.trade_id}</td>
      <td>${r.mode}</td>
      <td>${new Date(r.ts_et || r.ts_utc).toLocaleString()}</td>
      <td>${r.side}</td>
      <td>${r.ticker}</td>
      <td>${Number(r.qty||0).toFixed(6)}</td>
      <td>${Number(r.fill_price||0).toFixed(2)}</td>
      <td>${Number(r.notional||0).toFixed(2)}</td>
      <td>${Number(r.risk_frac||0).toFixed(3)}</td>
      <td>${Number(r.realized_pnl||0).toFixed(2)}</td>
      <td>${Number(r.equity_after||0).toFixed(2)}</td>
      <td>${r.reason||""}</td>
    `;
    tbody.appendChild(tr);
  }
}

// NEW: training stats (/api/stats)
async function loadStats() {
  const s = await getJSON("/stats");
  const train = s.state || {};
  const hist  = s.train || {};
  const reward = train.train_reward_mean ?? hist.last_reward_mean ?? 0;
  const win    = train.train_win_rate   ?? hist.last_win_rate   ?? 0;

  document.getElementById("rewardMean").textContent = Number(reward).toFixed(4);
  document.getElementById("winRate").textContent = `${(Number(win) * 100).toFixed(1)}%`;
  if (hist.last_time_utc) {
    document.getElementById("lastTrain").textContent = new Date(hist.last_time_utc).toLocaleString();
  }
}

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
