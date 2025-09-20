async function getJSON(path) {
  const headers = { "Content-Type": "application/json" };
  // If you don’t have a Netlify Function adding the Authorization header, do it here:
  if (window.API_KEY && window.API_KEY !== "<YOUR_API_KEY>") {
    headers["Authorization"] = `Bearer ${window.API_KEY}`;
  }
  const res = await fetch(`${window.API_BASE}${path}`, { headers, cache: "no-store" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

const fmtUSD = n => (n>=0? "$"+n.toFixed(2) : "-$"+Math.abs(n).toFixed(2));

let eqChart;

async function loadMetrics() {
  const m = await getJSON("/metrics");
  document.getElementById("mode").textContent = m.status || "idle";
  document.getElementById("equity").textContent = fmtUSD(Number(m.equity||0));
  document.getElementById("pnl").textContent = fmtUSD(Number(m.today_pnl||0));
  document.getElementById("trades").textContent = Number(m.today_trades||0);
  document.getElementById("universe").textContent = Number(m.universe||0);
  document.getElementById("dd").textContent = ((Number(m.max_drawdown||0))*100).toFixed(2)+"%";
}

async function loadEquity() {
  const data = await getJSON("/equity"); // defaults to current session
  const points = data.series || [];
  const labels = points.map(p => new Date(p.t));
  const values = points.map(p => Number(p.equity));

  if (!eqChart) {
    const ctx = document.getElementById("equityChart").getContext("2d");
    eqChart = new Chart(ctx, {
      type: "line",
      data: { labels, datasets: [{ label: "Equity", data: values, fill: false, tension: 0.2 }] },
      options: {
        parsing: false,
        animation: false,
        scales: {
          x: { type: "time", time: { unit: "minute" } },
          y: { beginAtZero: false }
        },
        plugins: { legend: { display: true } }
      }
    });
  } else {
    eqChart.data.labels = labels;
    eqChart.data.datasets[0].data = values;
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
      <td>${Number(r.qty||0).toFixed(4)}</td>
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
