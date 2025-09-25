const API_BASE = "/api";

// helpers
const fmtUSD = (n) => {
  const v = Number(n || 0);
  const s = `$${Math.abs(v).toFixed(2)}`;
  return v >= 0 ? s : `-${s}`;
};
const pct = (x) => `${(Number(x || 0) * 100).toFixed(2)}%`;

// generic fetch
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

// warm up
async function waitForApiReady() {
  for (let i = 0; i < 8; i++) {
    try { if ((await fetch(`${API_BASE}/healthz`)).ok) return; } catch {}
    await new Promise(r => setTimeout(r, 1000));
  }
}

// admin buttons
document.getElementById("resetBtn")?.addEventListener("click", async () => {
  const pwd = prompt("Admin password:");
  if (!pwd) return;
  try { await postJSON("/admin/reset", { password: pwd }); alert("Reset complete"); }
  catch (e) { alert(`Reset failed: ${e.message}`); }
});
document.getElementById("pauseBtn")?.addEventListener("click", async () => {
  const pwd = prompt("Admin password:");
  if (!pwd) return;
  try { await postJSON("/admin/pause", { password: pwd }); } catch(e){ alert(e.message); }
});
document.getElementById("resumeBtn")?.addEventListener("click", async () => {
  const pwd = prompt("Admin password:");
  if (!pwd) return;
  try { await postJSON("/admin/resume", { password: pwd }); } catch(e){ alert(e.message); }
});
document.getElementById("paramsBtn")?.addEventListener("click", async () => {
  try {
    const data = await getJSON("/stats");
    const box = document.getElementById("paramsBox");
    if (box) { box.style.display = "block"; box.textContent = JSON.stringify(data, null, 2); }
    else { alert(JSON.stringify(data, null, 2)); }
  } catch (e) { alert(e.message); }
});

// manual trade
document.getElementById("tradeBtn")?.addEventListener("click", async () => {
  const password = prompt("Admin password for manual trade:");
  if (!password) return;
  const side = document.getElementById("tradeSide").value;
  const ticker = document.getElementById("tradeTicker").value.trim().toUpperCase();
  const notional = parseFloat(document.getElementById("tradeNotional").value || "0");
  if (!ticker || !side) return alert("Provide side and ticker.");
  try {
    await postJSON("/admin/trade", { password, side, ticker, notional: isNaN(notional)?undefined:notional });
    alert("Order queued to next live tick.");
  } catch (e) { alert(e.message); }
});

// metrics
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

// equity (with cached benchmark every 60s)
let eqChart;
let lastBenchAt = 0;
async function loadEquity() {
  const wantBench = (Date.now() - lastBenchAt) > 60_000;
  const { series = [], bench = [] } = await getJSON(`/equity?bench=${wantBench ? 1 : 0}`);
  if (wantBench) lastBenchAt = Date.now();
  const equityPts = series.map(p => ({ x: p.t, y: Number(p.equity) }));
  const benchPts  = (bench || []).map(p => ({ x: p.t, y: Number(p.equity) }));
  if (!eqChart) {
    const ctx = document.getElementById("equityChart").getContext("2d");
    eqChart = new Chart(ctx, {
      type: "line",
      data: { datasets: [
        { label: "Equity", data: equityPts, tension: 0.25, pointRadius: 0 },
        { label: "Benchmark (SPY)", data: benchPts, tension: 0.25, pointRadius: 0, borderColor: "#ff6b6b" }
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
    if (benchPts.length) eqChart.data.datasets[1].data = benchPts;
    eqChart.update();
  }
}

// trades
let tradesCursor = 0;
async function loadTrades() {
  const payload = await getJSON(`/trades?limit=100&cursor=${tradesCursor}`);
  tradesCursor = payload.next_cursor || tradesCursor;
  const rows = payload.data || [];
  const tbody = document.getElementById("tradesBody");
  tbody.innerHTML = "";
  for (const r of rows) {
    const tr = document.createElement("tr");
    const pnl = Number(r.realized_pnl||0);
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
      <td class="${pnl>=0?'good':'bad'}">${pnl.toFixed(2)}</td>
      <td>${Number(r.equity_after||0).toFixed(2)}</td>
      <td>${r.reason||""}</td>
    `;
    tbody.appendChild(tr);
  }
}

// training stats
async function loadStats() {
  const s = await getJSON("/stats");
  const state = s.state || {}, train = s.train || {};
  const reward = state.train_reward_mean ?? train.last_reward_mean ?? 0;
  const win    = state.train_win_rate   ?? train.last_win_rate   ?? 0;
  document.getElementById("rewardMean").textContent = Number(reward).toFixed(4);
  document.getElementById("winRate").textContent = `${(Number(win) * 100).toFixed(1)}%`;
  document.getElementById("lastTrain").textContent =
    train.last_time_utc ? new Date(train.last_time_utc).toLocaleString() : "–";
}

// refresh loop
async function refresh() {
  try {
    await Promise.all([loadMetrics(), loadEquity(), loadTrades(), loadStats()]);
  } catch (e) {
    console.warn("UI refresh error:", e.message);
  } finally {
    setTimeout(refresh, 5000);
  }
}

// boot
(async () => { await waitForApiReady(); refresh(); })();
