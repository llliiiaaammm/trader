// netlify/functions/proxy.js
exports.handler = async (event) => {
  try {
    const backend = process.env.API_BASE_URL; // e.g. https://trader-xxxx.onrender.com
    const apiKey  = process.env.API_KEY;

    if (!backend || !apiKey) {
      return { statusCode: 500, body: "Missing API_BASE_URL or API_KEY" };
    }

    // Build target URL: strip /api prefix, keep query string
    const pathNoApi = event.path.replace(/^\/api/, "");
    const qs = event.rawQuery ? `?${event.rawQuery}` : "";
    const target = backend.replace(/\/+$/, "") + pathNoApi + qs;

    // Forward headers, but set Authorization and Host for upstream
    const headers = {};
    for (const [k, v] of Object.entries(event.headers || {})) {
      const lk = k.toLowerCase();
      if (lk !== "host" && lk !== "authorization") headers[k] = v;
    }
    headers["Authorization"] = `Bearer ${apiKey}`;
    headers["Host"] = new URL(backend).host;

    // Build fetch init; DO NOT send body for GET/HEAD
    const init = { method: event.httpMethod, headers };
    if (event.httpMethod !== "GET" && event.httpMethod !== "HEAD" && event.body) {
      init.body = event.isBase64Encoded ? Buffer.from(event.body, "base64") : event.body;
    }

    const upstream = await fetch(target, init);

    // Read upstream as text (covers JSON & text). Good enough for our API.
    const respText = await upstream.text();
    const respHeaders = {};
    upstream.headers.forEach((v, k) => (respHeaders[k] = v));

    return {
      statusCode: upstream.status,
      headers: respHeaders,
      body: respText,
    };
  } catch (err) {
    return { statusCode: 502, body: `Proxy error: ${err}` };
  }
};
