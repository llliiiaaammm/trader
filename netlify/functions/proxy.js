// Proxies /api/* to your Render backend.
// Netlify env (Site settings → Environment):
//   BASE_URL = https://<your-render>.onrender.com
//   API_KEY  = <same value as API_KEY on Render>

exports.handler = async (event) => {
  try {
    const backend = process.env.BASE_URL;
    const apiKey  = process.env.API_KEY;
    if (!backend || !apiKey) {
      return { statusCode: 500, body: "Missing BASE_URL or API_KEY" };
    }

    // With the :splat redirect, event.path is "/.netlify/functions/proxy/<splat>"
    const path = event.path.replace(/^\/\.netlify\/functions\/proxy/, "");
    const qs   = event.rawQuery ? `?${event.rawQuery}` : "";
    const target = backend.replace(/\/+$/, "") + path + qs;

    // Forward headers (no host/auth), add Authorization
    const headers = {};
    for (const [k, v] of Object.entries(event.headers || {})) {
      const lk = k.toLowerCase();
      if (lk !== "authorization" && lk !== "host") headers[k] = v;
    }
    headers["Authorization"]   = `Bearer ${apiKey}`;
    headers["Host"]            = new URL(backend).host;
    headers["Accept-Encoding"] = "identity"; // avoid compressed upstream body

    const init = { method: event.httpMethod, headers };
    if (!["GET", "HEAD"].includes(event.httpMethod) && event.body) {
      init.body = event.isBase64Encoded ? Buffer.from(event.body, "base64") : event.body;
    }

    const upstream = await fetch(target, init);
    const text = await upstream.text();

    // Strip hop-by-hop headers
    const respHeaders = {};
    upstream.headers.forEach((v, k) => (respHeaders[k] = v));
    delete respHeaders["content-encoding"]; delete respHeaders["Content-Encoding"];
    delete respHeaders["content-length"];   delete respHeaders["Content-Length"];

    return { statusCode: upstream.status, headers: respHeaders, body: text };
  } catch (err) {
    return { statusCode: 502, body: `Proxy error: ${err}` };
  }
};
