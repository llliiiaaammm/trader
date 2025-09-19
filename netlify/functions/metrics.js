export async function handler() {
  const API_BASE = process.env.API_BASE_URL;   // e.g. https://your-backend.onrender.com
  const API_KEY  = process.env.API_KEY;        // same value as on the backend
  const r = await fetch(`${API_BASE}/metrics`, {
    headers: { Authorization: `Bearer ${API_KEY}` }
  });
  return {
    statusCode: r.status,
    body: await r.text(),
    headers: { 'content-type': 'application/json' }
  };
}

