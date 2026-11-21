import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiKey = env.SKYGUARD_API_KEY

  if (!apiKey) {
    console.warn(
      'WARNING: SKYGUARD_API_KEY is missing. Protected API requests will fail.'
    )
  }

  return {
    plugins: [react()],
    server: {
      host: '127.0.0.1',
      port: 5173,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          headers: apiKey
            ? { 'X-API-Key': apiKey }
            : {},
        },
      },
    },
  }
})
