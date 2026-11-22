import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiKey = env.SKYGUARD_API_KEY
  const persistenceKey = command === 'serve' ? env.SKYGUARD_DB_API_KEY : undefined
  const backendPort = env.SKYGUARD_BACKEND_PORT || '8000'
  if (!/^\d{1,5}$/.test(backendPort) || Number(backendPort) < 1 || Number(backendPort) > 65535) {
    throw new Error('SKYGUARD_BACKEND_PORT must be a TCP port')
  }
  if (command === 'serve' && !apiKey) {
    console.warn('Set SKYGUARD_API_KEY in the Vite and backend process environments for protected media requests.')
  }

  return {
    define: {
      'import.meta.env.VITE_SKYGUARD_LOCAL_PROXY': JSON.stringify(Boolean(persistenceKey)),
      'import.meta.env.VITE_SKYGUARD_DEMO_MODE': JSON.stringify(command === 'serve' && env.SKYGUARD_DEMO_MODE === '1'),
    },
    plugins: [react(), {
      name: 'skyguard-loopback-persistence',
      configureServer(server) {
        if (!persistenceKey) return
        server.middlewares.use((request, response, next) => {
          if (!/^\/api\/persistence(?:\/|\?|$)/.test(request.url ?? '')) {
            next()
            return
          }
          const host = request.headers.host ?? ''
          const origin = request.headers.origin
          const fetchSite = request.headers['sec-fetch-site']
          const fromLoopback = ['127.0.0.1', '::1', '::ffff:127.0.0.1'].includes(request.socket.remoteAddress ?? '')
          const localHost = /^(?:127\.0\.0\.1|localhost|\[::1\])(?::\d+)?$/i.test(host)
          if (!fromLoopback || !localHost || (origin && origin !== 'http://' + host) ||
              (fetchSite && fetchSite !== 'same-origin' && fetchSite !== 'none')) {
            response.writeHead(403, {'Content-Type': 'application/json'})
            response.end(JSON.stringify({ detail: 'SkyGuard persistence requires a same-origin loopback request' }))
            return
          }
          request.headers.authorization = 'Bearer ' + persistenceKey
          next()
        })
      },
    }],
    server: {
      host: '127.0.0.1',
      port: 5173,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:' + backendPort,
          changeOrigin: true,
          headers: apiKey ? { 'X-API-Key': apiKey } : {},
        },
        '/ws': { target: 'ws://127.0.0.1:' + backendPort, ws: true },
      },
    },
    build: {
      rollupOptions: {
        input: {
          existing: fileURLToPath(new URL('./index.html', import.meta.url)),
          operator: fileURLToPath(new URL('./operator.html', import.meta.url)),
        },
      },
    },
  }
})
