// All frontend calls go through Netlify proxy as /api/*
// The proxy injects Authorization using Netlify env (API_KEY)
// and forwards to your Render BASE_URL.
const API_BASE = "/api";

// --- helpers
const fmtUSD = (n) => {
  const v = Number(n || 0);
  return v >= 0 ? `$${v.toFixed(2)}` : `-$${Math.abs(v).toFixed(2)}`;
};
const pct = (x) => `${(Number(x || 0) * 100).toFixed(2)}%`;

// --- generic fetch
async function getJSON(path) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function postJSON(path, body) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

// --- admin reset
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

// --- metrics
async function loadMetrics() {
  const m = await getJSON("/metrics");
  document.getElementById("mode").innerHTML = `<b>${m.status || "idle"}</b>`;
  document.getElementById("equity").textContent = fmtUSD(m.equity);
  document.getElementById("pnl").textContent = fmtUSD(m.today_pnl);
  document.getElementById("trades").textContent = Number(m.today_trades || 0);
  document.getElementById("universe").textContent = Number(m.universe || 0);
  document.getElementById("dd").textContent = pct(m.max_drawdown);

  // wallet block
  document.getElementById("cash").textContent = fmtUSD(m.cash);
  document.getElementById("invested").textContent = fmtUSD(m.positions_value);
  document.getElementById("upnl").textContent = fmtUSD(m.unrealized_pnl);
}

// --- equity chart
let eqChart;
async function loadEquity() {
  const { series = [] } = await getJSON("/equity");
  const points = series.map(p => ({ x: p.t, y: Number(p.equity) }));

  if (!eqChart) {
    const ctx = document.getElementById("equityChart").getContext("2d");
    eqChart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [{
          label: "Equity",
          data: points,
          tension: 0.2,
          pointRadius: 0,
          fill: false
        }]
      },
      options: {
        responsive: true,
        animation: false,
        parsing: true,      // parse {x,y}
        interaction: { mode: "nearest", intersect: false },
        scales: {
          x: {
            type: "time",
            time: { unit: "minute", tooltipFormat: "MMM d, HH:mm" }
          },
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

// --- trades table (paged by cursor; we show most recent page)
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

// --- training stats (/stats)
async function loadStats() {
  const s = await getJSON("/stats");
  const state = s.state || {};
  const train = s.train || {};

  const reward = state.train_reward_mean ?? train.last_reward_mean ?? 0;
  const win    = state.train_win_rate   ?? train.last_win_rate   ?? 0;

  document.getElementById("rewardMean").textContent = Number(reward).toFixed(4);
  document.getElementById("winRate").textContent = `${(Number(win) * 100).toFixed(1)}%`;
  document.getElementById("lastTrain").textContent =
    train.last_time_utc ? new Date(train.last_time_utc).toLocaleString() : "–";
}

// --- refresh loop
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
