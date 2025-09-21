console.log("DBG API_BASE bootstrap", { RAW_BASE: (window||{}).API_BASE, RAW_KEY: (window||{}).API_KEY });

// --- Base URL + auth (sanitized) ---
const RAW_BASE = (typeof window !== "undefined" ? window.API_BASE : null);
const API_BASE = (!RAW_BASE || RAW_BASE === "undefined" ? "/api" : RAW_BASE.trim().replace(/\/+$/, ""));
const RAW_KEY  = (typeof window !== "undefined" ? window.API_KEY : null);
const API_KEY  = (!RAW_KEY || RAW_KEY === "undefined" ? "" : RAW_KEY);

// single, final getJSON
async function getJSON(path) {
  const headers = { "Content-Type": "application/json" };
  if (API_KEY) headers["Authorization"] = `Bearer ${API_KEY}`;
  const url = `${API_BASE}${path}`;
  console.log("DBG fetch", { API_BASE, url, path });   // keep this log until fixed
  const res = await fetch(url, { headers, cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

async function postJSON(path, body) {
  const headers = { "Content-Type": "application/json" };
  if (API_KEY) headers["Authorization"] = `Bearer ${API_KEY}`;
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers,
    body: JSON.stringify(body || {})
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

document.getElementById("resetBtn")?.addEventListener("click", async () => {
  try {
    const pwd = prompt("Admin password to reset all model & trades:");
    if (!pwd) return;
    await postJSON("/admin/reset", { password: pwd });
    alert("Reset complete. The service will start a new episode block.");
  } catch (e) {
    alert(`Reset failed: ${e.message}`);
  }
});

const fmtUSD = n => (n>=0? "$"+n.toFixed(2) : "-$"+Math.abs(n).toFixed(2));

async function loadMetrics() {
  const m = await getJSON("/metrics");
  document.getElementById("mode").textContent = m.status || "idle";
  document.getElementById("equity").textContent = fmtUSD(Number(m.equity||0));
  document.getElementById("pnl").textContent = fmtUSD(Number(m.today_pnl||0));
  document.getElementById("trades").textContent = Number(m.today_trades||0);
  document.getElementById("universe").textContent = Number(m.universe||0);
  document.getElementById("dd").textContent = ((Number(m.max_drawdown||0))*100).toFixed(2)+"%";
}

let eqChart;

async function loadEquity() {
  // API returns: { series: [{ t: ISO_STRING, equity: number }], session: "..." }
  const { series = [] } = await getJSON("/equity");

  const points = series.map(p => ({
    x: p.t,                 // ISO string is fine; adapter parses it
    y: Number(p.equity)
  }));

  if (!eqChart) {
    const ctx = document.getElementById("equityChart").getContext("2d");
    eqChart = new Chart(ctx, {
      type: "line",
      data: {
        datasets: [{
          label: "Equity",
          data: points,
          fill: false,
          tension: 0.2,
          pointRadius: 0
        }]
      },
      options: {
        animation: false,
        parsing: true,      // enable {x,y} parsing
        scales: {
          x: {
            type: "time",
            time: {
              unit: "minute",          // 'hour' or 'day' if you prefer
              tooltipFormat: "MMM d, HH:mm"
            }
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

async function refresh() {
  try {
    await Promise.all([loadMetrics(), loadEquity(), loadTrades()]);
  } catch (e) {
    // keep UI running even if API hiccups (e.g., 504 due to cold start)
    console.warn(e);
  } finally {
    setTimeout(refresh, 5000);
  }
}

refresh();
