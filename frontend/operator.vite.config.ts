import {defineConfig} from 'vite'
import react from '@vitejs/plugin-react'
import {fileURLToPath} from 'node:url'

// Side-by-side operator console. Existing Vite config and existing App remain intact.
// The API proxy is deliberately localhost-only for local testing.
export default defineConfig({
 plugins:[react()],
 server:{host:'127.0.0.1',proxy:{'/api':{target:'http://127.0.0.1:8000',changeOrigin:true},'/ws':{target:'ws://127.0.0.1:8000',ws:true}}},
 build:{rollupOptions:{input:{existing:fileURLToPath(new URL('./index.html',import.meta.url)),operator:fileURLToPath(new URL('./operator.html',import.meta.url))}}}
})
