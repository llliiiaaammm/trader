// netlify/functions/proxy.js
exports.handler = async (event, context) => {
  const backend = process.env.API_BASE_URL;         // e.g. https://trader-xxxx.onrender.com
  const apiKey  = process.env.API_KEY;              // same secret as Render API_KEY
  if (!backend || !apiKey) {
    return { statusCode: 500, body: "Missing API_BASE_URL or API_KEY" };
  }

  const makeTarget = () => {
    const pathNoApi = event.path.replace(/^\/api/, "");
    const qs = event.rawQuery ? `?${event.rawQuery}` : "";
    return backend.replace(/\/+$/, "") + pathNoApi + qs;
  };

  // clone headers, inject auth, avoid gzip to keep lambdas small
  const buildInit = (methodOverride) => {
    const headers = {};
    for (const [k, v] of Object.entries(event.headers || {})) {
      const lk = k.toLowerCase();
      if (lk !== "host" && lk !== "authorization" && lk !== "content-length") headers[k] = v;
    }
    headers["Authorization"]   = `Bearer ${apiKey}`;
    headers["Host"]            = new URL(backend).host;
    headers["Accept-Encoding"] = "identity";

    const init = { method: methodOverride || event.httpMethod, headers };
    if (init.method !== "GET" && init.method !== "HEAD" && event.body) {
      init.body = event.isBase64Encoded ? Buffer.from(event.body, "base64") : event.body;
    }
    return init;
  };

  // 25s fetch timeout (your plan still limits overall lambda time; see netlify.toml)
  const fetchWithTimeout = async (url, init, ms = 25000) => {
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), ms);
    try {
      return await fetch(url, { ...init, signal: ctrl.signal });
    } finally {
      clearTimeout(t);
    }
  };

  const target = makeTarget();

  // Step 1: try once
  try {
    const up = await fetchWithTimeout(target, buildInit());
    if (up.status >= 200 && up.status < 500) {
      const text = await up.text();
      const headers = {};
      up.headers.forEach((v, k) => (headers[k] = v));
      delete headers["content-encoding"]; delete headers["Content-Encoding"];
      delete headers["content-length"];    delete headers["Content-Length"];
      return { statusCode: up.status, headers, body: text };
    }
    // fall through to warm + retry for 5xx
  } catch (_) {
    // fall through to warm + retry on network/cold errors
  }

  // Step 2: warm Render (healthz) and retry once after short wait
  try {
    await fetchWithTimeout(backend.replace(/\/+$/, "") + "/healthz", buildInit("GET"), 5000);
  } catch (_) { /* ignore */ }

  await new Promise(r => setTimeout(r, 1500));

  try {
    const up2 = await fetchWithTimeout(target, buildInit(), 25000);
    const text = await up2.text();
    const headers = {};
    up2.headers.forEach((v, k) => (headers[k] = v));
    delete headers["content-encoding"]; delete headers["Content-Encoding"];
    delete headers["content-length"];    delete headers["Content-Length"];
    return { statusCode: up2.status, headers, body: text };
  } catch (err) {
    return { statusCode: 504, body: `Gateway Timeout via proxy: ${err}` };
  }
};
