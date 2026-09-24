/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8000/api/:path*',
      },
    ];
  },
  // Increase the server-action / API route timeout to 5 minutes
  // so long-running ingest requests (scraping + embedding) don't get cut off
  experimental: {
    proxyTimeout: 180000, // 3 minutes for long-running ingest scraping
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  // Proxy timeout configuration (ms) — allow up to 5 min for ingest
  httpAgentOptions: {
    keepAlive: true,
  },
};

module.exports = nextConfig;
