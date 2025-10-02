/* ---------- RUNTIME API BASE DETECTION ---------- */
let API_BASE = "/api";     // default for Netlify-style proxies
let _ready = (async () => {
  const candidates = [
    localStorage.getItem("apiBase") || "",  // user override ('' == root)
    "/api",
    ""
  ];

  async function ok(base) {
    try {
      const r = await fetch(`${base}/healthz`, { method: "GET", cache: "no-store" });
      return r.ok;
    } catch { return false; }
  }

  for (const base of candidates) {
    if (await ok(base)) { API_BASE = base; localStorage.setItem("apiBase", base); break; }
  }

  // If still unreachable, show a banner and keep trying root (so nothing crashes).
  try {
    if (!(await ok(API_BASE))) {
      console.error("Backend unreachable. Tried bases:", candidates);
      const note = document.createElement("div");
      note.style.cssText = "position:fixed;top:8px;left:8px;z-index:9999;background:#5b0000;color:#fff;padding:8px 10px;border-radius:8px;font:12px/1.2 system-ui";
      note.textContent = "Backend unreachable. Open DevTools > Console for details.";
      document.body.appendChild(note);
    }
  } catch {}
})();

/* ---------- SMALL HELPERS ---------- */
const fmtUSD = (n) => {
  const v = Number(n || 0);
  const s = `$${Math.abs(v).toFixed(2)}`;
  return v >= 0 ? s : `-${s}`;
};
const pct = (x) => `${(Number(x || 0) * 100).toFixed(2)}%`;
const r6 = (x) => Number(x || 0).toFixed(6);

async function getJSON(path) {
  await _ready;
  const url = `${API_BASE}${path}`;
  // one quick retry to smooth cold starts
  for (let i = 0; i < 1; i++) {
    try {
      const r = await fetch(url, { cache: "no-store" });
      if (r.ok) return r.json();
    } catch {}
    await new Promise((r) => setTimeout(r, 250));
  }
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} on ${url}`);
  return res.json();
}

async function postJSON(path, body) {
  await _ready;
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {})
  });
  if (!res.ok) throw new Error(`${res.status}`);
  return res.json();
}

/* ---------- HIDE THE BUTTONS YOU ASKED TO REMOVE ---------- */
["pauseBtn", "resumeBtn", "manualBtn", "tradeBtn", "tradeBox"].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.style.display = "none";
});

/* ---------- ADMIN: KEEP RESET ONLY ---------- */
document.getElementById("resetBtn")?.addEventListener("click", async () => {
  const pwd = prompt("Admin password to reset all model & trades:");
  if (!pwd) return;
  try { await postJSON("/admin/reset", { password: pwd }); alert("Reset complete."); }
  catch (e) { alert(`Reset failed: ${e.message}`); }
});

/* ---------- BENCHMARK PICKER (CLICK THE LABEL) ---------- */
const BENCH_KEY = "benchSymbol";
const benchSymbol = () => localStorage.getItem(BENCH_KEY) || "SPY";
const setBench = (s) => localStorage.setItem(BENCH_KEY, (s || "SPY").toUpperCase());
document.getElementById("benchLabel")?.addEventListener("click", async () => {
  const sym = prompt("Benchmark ticker (e.g., SPY, QQQ):", benchSymbol());
  if (!sym) return;
  setBench(sym);
  const el = document.getElementById("benchLabel");
  if (el) el.textContent = benchSymbol();
  await loadEquity();
});

/* ---------- METRICS PANEL ---------- */
async function loadMetrics() {
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
}

/* ---------- EQUITY + BENCH CHART ---------- */
let eqChart;
async function loadEquity() {
  const bench = benchSymbol();
  const payload = await getJSON(`/equity?bench=${encodeURIComponent(bench)}`);
  const { series = [], bench: benchSeries = [] } = payload;

  const equityPts = series.map(p => ({ x: p.t, y: Number(p.equity) }));
  const benchPts  = benchSeries.map(p => ({ x: p.t, y: Number(p.equity) }));

  const bEl = document.getElementById("benchLabel");
  if (bEl) bEl.textContent = bench;

  // Graceful if Chart.js isn't on the page
  if (typeof Chart === "undefined") return;

  const ctx = document.getElementById("equityChart")?.getContext?.("2d");
  if (!ctx) return;

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
        responsive: true,
        animation: false,
        parsing: true,
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
}

/* ---------- TRADES TABLE ---------- */
let tradesCursor = 0;
async function loadTrades() {
  const payload = await getJSON(`/trades?limit=100&cursor=${tradesCursor}`);
  tradesCursor = payload.next_cursor || tradesCursor;
  const rows = payload.data || [];
  const tbody = document.getElementById("tradesBody");
  if (!tbody) return;
  tbody.innerHTML = "";
  for (const r of rows) {
    const showPnl = (r.side || "").toUpperCase() === "SELL";   // PnL only on sells
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
      <td class="${showPnl ? (pnl>=0?'good':'bad') : ''}">${showPnl ? pnl.toFixed(2) : ""}</td>
      <td>${Number(r.equity_after||0).toFixed(2)}</td>
      <td>${r.reason||""}</td>`;
    tbody.appendChild(tr);
  }
}

/* ---------- TRAINING STATS ---------- */
async function loadStats() {
  const s = await getJSON("/stats");
  const train = s.train || {};
  document.getElementById("rewardMean").textContent = Number(train.last_reward_mean || 0).toFixed(4);
  document.getElementById("winRate").textContent = `${(Number(train.last_win_rate || 0) * 100).toFixed(1)}%`;
  document.getElementById("lastTrain").textContent = train.last_time_utc ? new Date(train.last_time_utc).toLocaleString() : "–";
}

/* ---------- POLLING (staggered) ---------- */
(async () => {
  try {
    await _ready;
    await loadMetrics();
    await loadEquity();
    await loadTrades();
    await loadStats();

    setInterval(() => loadMetrics().catch(e => console.warn("metrics", e.message)), 5000);
    setInterval(() => loadTrades().catch(e => console.warn("trades", e.message)), 10000);
    setInterval(() => loadEquity().catch(e => console.warn("equity", e.message)), 30000);
    setInterval(() => loadStats().catch(e => console.warn("stats", e.message)), 30000);
  } catch (e) {
    console.warn("Initial UI load error:", e.message);
  }
})();
