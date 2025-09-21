export async function onRequest(context) {
  const { request, env } = context;

  const backend = env.API_BASE_URL; // e.g. https://trader-xxxx.onrender.com
  const apiKey  = env.API_KEY;      // same key as your backend expects

  const incoming = new URL(request.url);
  // Strip the /api prefix
  const path = incoming.pathname.replace(/^\/api/, "");
  const target = backend.replace(/\/+$/, "") + path + incoming.search;

  const headers = new Headers(request.headers);
  headers.set("Authorization", `Bearer ${apiKey}`);
  headers.set("Host", new URL(backend).host);

  const upstream = await fetch(target, {
    method: request.method,
    headers,
    body: request.body,
  });

  // pass through response
  const respHeaders = new Headers(upstream.headers);
  return new Response(upstream.body, { status: upstream.status, headers: respHeaders });
}
