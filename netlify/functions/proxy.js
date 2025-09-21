// netlify/functions/proxy.js
exports.handler = async (event) => {
  try {
    const backend = process.env.API_BASE_URL; // e.g. https://trader-xxxx.onrender.com
    const apiKey  = process.env.API_KEY;      // same key your backend expects

    if (!backend || !apiKey) {
      return { statusCode: 500, body: "Missing API_BASE_URL or API_KEY env var" };
    }

    // incoming path: /.netlify/functions/proxy OR /api/...
    // We redirect /api/* to this function, so event.path starts with /api/...
    const pathNoApi = event.path.replace(/^\/api/, ""); // keep leading slash for backend
    const qs = event.rawQuery ? `?${event.rawQuery}` : "";
    const target = backend.replace(/\/+$/, "") + pathNoApi + qs;

    // Build headers for upstream
    const headers = new Headers();
    for (const [k, v] of Object.entries(event.headers || {})) {
      if (!["host", "authorization"].includes(k.toLowerCase())) headers.set(k, v);
    }
    headers.set("Authorization", `Bearer ${apiKey}`);
    headers.set("Host", new URL(backend).host);

    const upstream = await fetch(target, {
      method: event.httpMethod,
      headers,
      body: event.body,
    });

    // Stream back the response (binary-safe)
    const buf = await upstream.arrayBuffer();
    const base64 = Buffer.from(buf).toString("base64");
    const respHeaders = {};
    upstream.headers.forEach((v, k) => (respHeaders[k] = v));

    return {
      statusCode: upstream.status,
      headers: respHeaders,
      body: base64,
      isBase64Encoded: true,
    };
  } catch (err) {
    return { statusCode: 502, body: `Proxy error: ${err}` };
  }
};
