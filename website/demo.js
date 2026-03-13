const API_BASE_URL = (window.API_BASE_URL || 'http://localhost:5000').replace(/\/$/, '');

// ── Client-side QAOA simulation (p=1, exact statevector) ────────────────
// Runs entirely in the browser when the API is unavailable (e.g. GitHub Pages).

function computeCutDiag(n, edges) {
  const num = 1 << n;
  const diag = new Float64Array(num);
  for (let s = 0; s < num; s++) {
    let v = 0;
    for (const [u, w] of edges) if (((s >> u) & 1) !== ((s >> w) & 1)) v++;
    diag[s] = v;
  }
  return diag;
}

function applyPhase(re, im, diag, gamma) {
  for (let s = 0; s < re.length; s++) {
    const a = -gamma * diag[s];
    const c = Math.cos(a), si = Math.sin(a);
    const r = re[s], i = im[s];
    re[s] = r * c - i * si;
    im[s] = r * si + i * c;
  }
}

function applyMixer(re, im, n, beta) {
  const cb = Math.cos(beta), sb = Math.sin(beta);
  for (let q = 0; q < n; q++) {
    const bit = 1 << q;
    for (let s = 0; s < re.length; s++) {
      if ((s & bit) === 0) {
        const s2 = s | bit;
        const r1 = re[s], i1 = im[s], r2 = re[s2], i2 = im[s2];
        re[s]  = cb * r1 + sb * i2;
        im[s]  = cb * i1 - sb * r2;
        re[s2] = cb * r2 + sb * i1;
        im[s2] = cb * i2 - sb * r1;
      }
    }
  }
}

function qaojsExpCut(n, edges, gamma, beta) {
  const num  = 1 << n;
  const norm = 1 / Math.sqrt(num);
  const re   = new Float64Array(num).fill(norm);
  const im   = new Float64Array(num).fill(0);
  const diag = computeCutDiag(n, edges);
  applyPhase(re, im, diag, gamma);
  applyMixer(re, im, n, beta);
  let cut = 0;
  for (let s = 0; s < num; s++) cut += (re[s] ** 2 + im[s] ** 2) * diag[s];
  return cut;
}

function qaojsOptimize(n, edges) {
  // Grid search over (gamma, beta) for p=1
  const steps = 24;
  let bestGamma = Math.PI / 4, bestBeta = Math.PI / 8, bestCut = -Infinity;
  for (let gi = 0; gi < steps; gi++) {
    for (let bi = 0; bi < steps; bi++) {
      const gamma = (gi + 0.5) * Math.PI / steps;
      const beta  = (bi + 0.5) * Math.PI / (2 * steps);
      const cut   = qaojsExpCut(n, edges, gamma, beta);
      if (cut > bestCut) { bestCut = cut; bestGamma = gamma; bestBeta = beta; }
    }
  }
  return { gammas: [bestGamma], betas: [bestBeta], expected_cut: bestCut };
}
// ─────────────────────────────────────────────────────────────────────────────

const svg      = document.getElementById('graph-svg');
const rangeEl  = document.getElementById('n-range');
const nValEl   = document.getElementById('n-val');
const nInput   = document.getElementById('n');
const hintEl   = document.getElementById('graph-hint');

const resultPanel = document.getElementById('result-panel');
const errorPanel  = document.getElementById('error-panel');
const resGammas   = document.getElementById('res-gammas');
const resBetas    = document.getElementById('res-betas');
const resCut      = document.getElementById('res-cut');
const errorMsg    = document.getElementById('error-msg');

let currentEdges = [];
let currentN = 6;

// Sync slider → badge + hidden input
rangeEl.addEventListener('input', () => {
  currentN = parseInt(rangeEl.value);
  nValEl.textContent = currentN;
  nInput.value = currentN;
});

// ── Graph rendering ──────────────────────────────────────
function renderGraph(n, edges) {
  svg.innerHTML = '';
  const W = svg.clientWidth || 480;
  const H = 300;
  const cx = W / 2, cy = H / 2;
  const r  = Math.min(W, H) * 0.36;

  // Node positions on a circle
  const pos = Array.from({length: n}, (_, i) => ({
    x: cx + r * Math.cos((2 * Math.PI * i / n) - Math.PI / 2),
    y: cy + r * Math.sin((2 * Math.PI * i / n) - Math.PI / 2)
  }));

  // Edges
  const edgeGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  edges.forEach(([u, v]) => {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', pos[u].x);
    line.setAttribute('y1', pos[u].y);
    line.setAttribute('x2', pos[v].x);
    line.setAttribute('y2', pos[v].y);
    line.setAttribute('stroke', '#3d4170');
    line.setAttribute('stroke-width', '1.5');
    edgeGroup.appendChild(line);
  });
  svg.appendChild(edgeGroup);

  // Nodes
  pos.forEach((p, i) => {
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');

    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', p.x);
    circle.setAttribute('cy', p.y);
    circle.setAttribute('r', 16);
    circle.setAttribute('fill', '#312e81');
    circle.setAttribute('stroke', '#6366f1');
    circle.setAttribute('stroke-width', '1.5');
    g.appendChild(circle);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', p.x);
    text.setAttribute('y', p.y + 5);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', '#c7d2fe');
    text.setAttribute('font-size', '12');
    text.setAttribute('font-family', 'Inter, sans-serif');
    text.setAttribute('font-weight', '500');
    text.textContent = i;
    g.appendChild(text);

    svg.appendChild(g);
  });

  hintEl.textContent = edges.length
    ? `${n} nodes · ${edges.length} edges`
    : `${n} nodes · no edges yet — click Random Graph`;
}

// ── Button handlers ──────────────────────────────────────
document.getElementById('random').addEventListener('click', () => {
  currentN = parseInt(rangeEl.value);
  // Ensure at least one edge; retry until connected-enough
  do {
    currentEdges = [];
    for (let i = 0; i < currentN; i++)
      for (let j = i + 1; j < currentN; j++)
        if (Math.random() < 0.45) currentEdges.push([i, j]);
  } while (currentEdges.length === 0);

  resultPanel.classList.add('hidden');
  errorPanel.classList.add('hidden');
  renderGraph(currentN, currentEdges);
});

document.getElementById('predict').addEventListener('click', async () => {
  if (currentEdges.length === 0) {
    showError('Generate a random graph first.');
    return;
  }
  errorPanel.classList.add('hidden');

  let data = null;
  let source = 'api';

  // 1. Try the local Flask API
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({n: currentN, edges: currentEdges})
    });
    if (!res.ok) throw new Error(`${res.status}`);
    data = await res.json();
  } catch (_) {
    // 2. API unavailable — run QAOA entirely in the browser
    data   = qaojsOptimize(currentN, currentEdges);
    source = 'browser';
  }

  resGammas.textContent = data.gammas.map(v => v.toFixed(4)).join(', ');
  resBetas.textContent  = data.betas.map(v  => v.toFixed(4)).join(', ');
  resCut.textContent    = data.expected_cut.toFixed(4);

  const badge = document.getElementById('source-badge');
  if (source === 'browser') {
    badge.textContent = '⚡ computed in browser (p=1 grid search)';
    badge.className = 'source-badge browser';
  } else {
    badge.textContent = '🖥 computed by local API (GNN)';
    badge.className = 'source-badge api';
  }
  badge.classList.remove('hidden');

  resultPanel.classList.remove('hidden');
  highlightCut(currentN, currentEdges, data.gammas, data.betas);
});

function showError(html) {
  errorMsg.innerHTML = html;
  errorPanel.classList.remove('hidden');
  resultPanel.classList.add('hidden');
}

function highlightCut(n, edges, gammas, betas) {
  // Simple visual: randomly colour approx half-edges as "cut" for illustration
  const lines = svg.querySelectorAll('line');
  lines.forEach((line, idx) => {
    const isCut = (idx + Math.round(gammas[0] * 10)) % 2 === 0;
    line.setAttribute('stroke', isCut ? '#6366f1' : '#2d3148');
    line.setAttribute('stroke-width', isCut ? '2' : '1.5');
  });
}

// Initial render
renderGraph(currentN, currentEdges);
