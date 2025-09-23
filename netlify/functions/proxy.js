// netlify/functions/proxy.js
exports.handler = async (event) => {
  try {
    // Accept either name to avoid env drift
    const backend =
      process.env.BASE_URL ||
      process.env.API_BASE_URL ||      // your current name in Netlify
      process.env.VITE_API_BASE_URL || // just in case
      "";

    const apiKey =
      process.env.API_KEY ||
      process.env.VITE_API_KEY || "";

    if (!backend || !apiKey) {
      return {
        statusCode: 500,
        body: `Missing backend or apiKey. has_BASE_URL=${Boolean(process.env.BASE_URL)} has_API_BASE_URL=${Boolean(process.env.API_BASE_URL)} has_API_KEY=${Boolean(apiKey)}`
      };
    }

    // Build URL
    const pathNoApi = event.path.replace(/^\/api/, "");
    const qs = event.rawQuery ? `?${event.rawQuery}` : "";
    const target = backend.replace(/\/+$/, "") + pathNoApi + qs;

    // Forward headers, strip host/authorization
    const headers = {};
    for (const [k, v] of Object.entries(event.headers || {})) {
      const lk = k.toLowerCase();
      if (lk !== "host" && lk !== "authorization") headers[k] = v;
    }
    headers["Authorization"]   = `Bearer ${apiKey}`;
    headers["Host"]            = new URL(backend).host;
    headers["Accept-Encoding"] = "identity"; // avoid gzipped upstream to simplify

    const init = { method: event.httpMethod, headers };
    if (event.httpMethod !== "GET" && event.httpMethod !== "HEAD" && event.body) {
      init.body = event.isBase64Encoded ? Buffer.from(event.body, "base64") : event.body;
    }

    const upstream = await fetch(target, init);
    const respText = await upstream.text();
    const respHeaders = {};
    upstream.headers.forEach((v, k) => (respHeaders[k] = v));
    delete respHeaders["content-encoding"];
    delete respHeaders["Content-Encoding"];
    delete respHeaders["content-length"];
    delete respHeaders["Content-Length"];

    return { statusCode: upstream.status, headers: respHeaders, body: respText };
  } catch (err) {
    return { statusCode: 502, body: `Proxy error: ${err}` };
  }
};
