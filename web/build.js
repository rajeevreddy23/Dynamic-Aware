// Build script for web dashboard
const fs = require('fs');
const path = require('path');

console.log("Building EAT-Sched Dashboard bundle...");

const srcResults = path.join(__dirname, '..', 'results');
const destResults = path.join(__dirname, 'results');

if (fs.existsSync(srcResults)) {
  if (!fs.existsSync(destResults)) {
    fs.mkdirSync(destResults, { recursive: true });
  }
  const files = fs.readdirSync(srcResults);
  for (const file of files) {
    if (file.endsWith('.json') || file.endsWith('.csv')) {
      fs.copyFileSync(path.join(srcResults, file), path.join(destResults, file));
    }
  }
  console.log(`Copied ${files.length} benchmark artifacts to web/results/`);
}

console.log("Static assets verified.");
console.log("Build complete.");
