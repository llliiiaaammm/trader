// netlify/functions/proxy.js
exports.handler = async (event) => {
  try {
    const backend = process.env.API_BASE_URL; // https://<your-render>.onrender.com
    const apiKey  = process.env.API_KEY;
    if (!backend || !apiKey) {
      return { statusCode: 500, body: "Missing API_BASE_URL or API_KEY" };
    }

    // Build target URL (strip /api prefix, keep query)
    const pathNoApi = event.path.replace(/^\/api/, "");
    const qs = event.rawQuery ? `?${event.rawQuery}` : "";
    const target = backend.replace(/\/+$/, "") + pathNoApi + qs;

    // Forward headers, inject Authorization, force identity (no gzip)
    const headers = {};
    for (const [k, v] of Object.entries(event.headers || {})) {
      const lk = k.toLowerCase();
      if (lk !== "host" && lk !== "authorization") headers[k] = v;
    }
    headers["Authorization"]   = `Bearer ${apiKey}`;
    headers["Host"]            = new URL(backend).host;
    headers["Accept-Encoding"] = "identity";   // ← no compression from upstream

    // Build fetch init (no body for GET/HEAD)
    const init = { method: event.httpMethod, headers };
    if (event.httpMethod !== "GET" && event.httpMethod !== "HEAD" && event.body) {
      init.body = event.isBase64Encoded ? Buffer.from(event.body, "base64") : event.body;
    }

    const upstream = await fetch(target, init);

    // Read as text (JSON/text) and strip compression headers
    const respText = await upstream.text();
    const respHeaders = {};
    upstream.headers.forEach((v, k) => (respHeaders[k] = v));
    delete respHeaders["content-encoding"];
    delete respHeaders["Content-Encoding"];
    delete respHeaders["content-length"];
    delete respHeaders["Content-Length"];

    return {
      statusCode: upstream.status,
      headers: respHeaders,
      body: respText,
    };
  } catch (err) {
    return { statusCode: 502, body: `Proxy error: ${err}` };
  }
};
