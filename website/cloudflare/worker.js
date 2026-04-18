/**
 * Ultron API Worker (ultron-api)
 * Separate from the routing worker (ultron-brain CF Worker).
 * Proxies /api/* requests from the website to the HF Space Brain.
 *
 * Deploy: cd website/cloudflare && npx wrangler deploy
 * Secrets: npx wrangler secret put AUTH_TOKEN
 */

const DEFAULT_BRAIN_URL = 'https://ghostdrive1-ultron1.hf.space';

const ALLOWED_ORIGINS = [
  'http://localhost:5173',
  'http://localhost:4173',
  'https://ultron.pages.dev',
  'https://ultron-control.pages.dev',
];

function corsHeaders(origin) {
  const allowed = ALLOWED_ORIGINS.includes(origin) ? origin : ALLOWED_ORIGINS[0];
  return {
    'Access-Control-Allow-Origin': allowed,
    'Access-Control-Allow-Methods': 'GET, POST, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    'Access-Control-Max-Age': '86400',
  };
}

function addCors(response, origin) {
  const headers = new Headers(response.headers);
  for (const [k, v] of Object.entries(corsHeaders(origin))) headers.set(k, v);
  return new Response(response.body, { status: response.status, headers });
}

async function checkAuth(request, env) {
  const auth = request.headers.get('Authorization') ?? '';
  const token = auth.startsWith('Bearer ') ? auth.slice(7) : auth;
  return token === (env.AUTH_TOKEN ?? '');
}

async function proxyToBrain(path, request, env) {
  const brainUrl = env.BRAIN_URL ?? DEFAULT_BRAIN_URL;
  const body = ['POST', 'PUT', 'PATCH'].includes(request.method) ? await request.text() : undefined;

  const res = await fetch(`${brainUrl}${path}`, {
    method: request.method,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${env.AUTH_TOKEN ?? ''}`,
    },
    body,
  });

  return new Response(await res.text(), {
    status: res.status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function stub(data) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });
}

export default {
  async fetch(request, env) {
    const origin = request.headers.get('Origin') ?? '';
    const { pathname } = new URL(request.url);

    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(origin) });
    }

    const authed = await checkAuth(request, env);
    if (!authed) {
      return addCors(new Response('Unauthorized', { status: 401 }), origin);
    }

    let response;

    try {
      if (pathname === '/api/health') {
        response = await proxyToBrain('/health', request, env);

      } else if (pathname === '/api/infer') {
        response = await proxyToBrain('/infer', request, env);

      } else if (pathname === '/api/sentinel/event') {
        response = await proxyToBrain('/sentinel/event', request, env);

      } else if (pathname === '/api/keys') {
        // TODO: implement key management endpoints in Brain (main.py)
        response = stub({ keys: [], message: 'Key management endpoint pending — add to Brain main.py' });

      } else if (pathname.startsWith('/api/memory/stm/')) {
        // TODO: implement STM viewer endpoint in Brain
        response = stub({ messages: [], message: 'STM viewer endpoint pending — add to Brain main.py' });

      } else if (pathname === '/api/sentinel/reports') {
        // TODO: fetch from Notion/Supabase
        response = stub({ reports: [], message: 'Sentinel reports endpoint pending' });

      } else if (pathname === '/api/projects') {
        // TODO: fetch from Supabase
        response = stub({ projects: [], message: 'Projects endpoint pending' });

      } else {
        response = new Response(JSON.stringify({ error: 'Not found' }), {
          status: 404,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    } catch (err) {
      response = new Response(JSON.stringify({ error: err.message }), {
        status: 502,
        headers: { 'Content-Type': 'application/json' },
      });
    }

    return addCors(response, origin);
  },
};
