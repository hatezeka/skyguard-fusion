import { fileURLToPath } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ command, mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiKey = env.SKYGUARD_API_KEY
  const backendPort = env.SKYGUARD_BACKEND_PORT || '8000'
  if (!/^\d{1,5}$/.test(backendPort) || Number(backendPort) < 1 || Number(backendPort) > 65535) {
    throw new Error('SKYGUARD_BACKEND_PORT must be a TCP port')
  }
  if (command === 'serve' && !apiKey) {
    console.warn('Set SKYGUARD_API_KEY in the Vite and backend process environments for protected media requests.')
  }

  return {
    plugins: [react()],
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
