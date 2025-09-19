export async function handler(event) {
  const API_BASE = process.env.API_BASE_URL;
  const API_KEY  = process.env.API_KEY;
  const qs = event.rawQuery ? `?${event.rawQuery}` : '';
  const r = await fetch(`${API_BASE}/equity${qs}`, {
    headers: { Authorization: `Bearer ${API_KEY}` }
  });
  return {
    statusCode: r.status,
    body: await r.text(),
    headers: { 'content-type': 'application/json' }
  };
}

