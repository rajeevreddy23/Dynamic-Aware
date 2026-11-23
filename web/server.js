const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 3000;
const ROOT_DIR = __dirname;
const RESULTS_DIR = path.join(ROOT_DIR, '..', 'results');

const server = http.createServer((req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  const url = req.url.split('?')[0];

  if (url === '/' || url === '/index.html') {
    const indexPath = path.join(ROOT_DIR, 'index.html');
    if (fs.existsSync(indexPath)) {
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      fs.createReadStream(indexPath).pipe(res);
      return;
    }
  }

  // Static results endpoint: lists and serves saved results/*.json
  if (url.startsWith('/results/')) {
    const filename = path.basename(url);
    const filePath = path.join(RESULTS_DIR, filename);
    if (fs.existsSync(filePath)) {
      const ext = path.extname(filename);
      const contentType = ext === '.json' ? 'application/json' : 'text/plain';
      res.writeHead(200, { 'Content-Type': contentType });
      fs.createReadStream(filePath).pipe(res);
      return;
    } else {
      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: `File not found: ${filename}` }));
      return;
    }
  }

  if (url === '/api/static-results') {
    try {
      if (fs.existsSync(RESULTS_DIR)) {
        const files = fs.readdirSync(RESULTS_DIR).filter(f => f.endsWith('.json'));
        const data = {};
        for (const file of files) {
          const content = fs.readFileSync(path.join(RESULTS_DIR, file), 'utf-8');
          try {
            data[file] = JSON.parse(content);
          } catch (e) {
            data[file] = null;
          }
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ available_files: files, results: data }));
        return;
      } else {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ available_files: [], results: {} }));
        return;
      }
    } catch (err) {
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: err.message }));
      return;
    }
  }

  // Serve static assets if any
  const potentialFile = path.join(ROOT_DIR, url);
  if (fs.existsSync(potentialFile) && fs.statSync(potentialFile).isFile()) {
    res.writeHead(200);
    fs.createReadStream(potentialFile).pipe(res);
    return;
  }

  res.writeHead(404, { 'Content-Type': 'text/plain' });
  res.end('Not Found');
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`EAT-Sched Dashboard server running on http://localhost:${PORT}`);
});
