import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // Expose to network for cluster access
    port: 3000,
    open: false  // Don't try to open browser on cluster
  }
})
