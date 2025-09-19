const q = (s)=>document.querySelector(s);
let equityChart, pnlChart;

async function getJSON(url){
  const r=await fetch(url);
  if(!r.ok) throw new Error(await r.text());
  return r.json();
}

async function loadMetrics(){
  const m = await getJSON('/api/metrics');
  q('#status').textContent = m.status;
  q('#equity').textContent = `$${m.equity.toFixed(2)}`;
  q('#pnl').textContent = `$${m.today_pnl.toFixed(2)}`;
  q('#trades').textContent = m.today_trades;
  q('#uni').textContent = m.universe;
  q('#dd').textContent = `${(m.max_drawdown*100).toFixed(2)}%`;
}

async function loadEquity(){
  const e = await getJSON('/api/equity?window=30d');
  const labels = e.series.map(p=>new Date(p.t));
  const values = e.series.map(p=>p.equity);

  if (!equityChart){
    equityChart = new Chart(q('#equityChart'), {
      type:'line',
      data:{ labels, datasets:[{ label:'Equity', data:values }] },
      options:{ animation:false, parsing:false, scales:{ x:{ type:'time', time:{ unit:'day' } } } }
    });
  } else {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = values;
    equityChart.update('none');
  }

  const pnl = values.map((v,i)=> i? v-values[i-1] : 0);
  if (!pnlChart){
    pnlChart = new Chart(q('#pnlChart'), {
      type:'bar',
      data:{ labels, datasets:[{ label:'PnL', data:pnl }] },
      options:{ animation:false, parsing:false, scales:{ x:{ display:false } } }
    });
  } else {
    pnlChart.data.labels=labels;
    pnlChart.data.datasets[0].data=pnl;
    pnlChart.update('none');
  }
}

function rowHTML(t){
  return `<tr>
    <td>${t.trade_id}</td>
    <td>${t.mode}</td>
    <td>${t.ts_et}</td>
    <td>${t.side}</td>
    <td>${t.ticker}</td>
    <td>${(t.qty||0).toFixed(4)}</td>
    <td>${(t.fill_price||0).toFixed(2)}</td>
    <td>${(t.notional||0).toFixed(2)}</td>
    <td>${(t.fees_bps||0)}</td>
    <td>${(t.risk_frac||0)}</td>
    <td>${(t.realized_pnl||0).toFixed(2)}</td>
    <td>${((t.realized_pnl_pct||0)*100).toFixed(2)}%</td>
    <td>${(t.equity_after||0).toFixed(2)}</td>
    <td>${t.reason||''}</td>
  </tr>`;
}

let tradesCursor = 0;
async function loadTrades(){
  const ticker = q('#tickerFilter').value.trim().toUpperCase();
  const mode = q('#modeFilter').value;
  const res = await getJSON(`/api/trades?limit=200&cursor=${tradesCursor}`);
  tradesCursor = res.next_cursor || tradesCursor;
  let rows = res.data;
  if (ticker) rows = rows.filter(r=>r.ticker===ticker);
  if (mode) rows = rows.filter(r=>r.mode===mode);
  q('#tradesTable tbody').innerHTML = rows.map(rowHTML).join('');
}

async function refresh(){
  try { await loadMetrics(); await loadEquity(); await loadTrades(); } catch(e){ console.error(e); }
}

q('#reload').addEventListener('click', ()=>{ tradesCursor=0; refresh(); });
setInterval(refresh, 5000);
refresh();

