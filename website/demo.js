function normalizeBaseUrl(url) {
  return String(url || '').trim().replace(/\/$/, '');
}

function buildApiCandidates() {
  const candidates = [];
  const configured = normalizeBaseUrl(window.API_BASE_URL);
  if (configured) candidates.push(configured);

  if (window.location.protocol.startsWith('http')) {
    const host = window.location.hostname;
    if (host && host !== 'localhost' && host !== '127.0.0.1' && !host.endsWith('github.io')) {
      candidates.push(`${window.location.protocol}//${host}:5000`);
    }
    candidates.push(`${window.location.protocol}//localhost:5000`);
    candidates.push(`${window.location.protocol}//127.0.0.1:5000`);
  }

  candidates.push('http://localhost:5000');
  candidates.push('http://127.0.0.1:5000');

  return [...new Set(candidates.filter(Boolean).map(normalizeBaseUrl))];
}

const API_BASE_CANDIDATES = buildApiCandidates();

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

function qaojsExpCutFromDiag(n, diag, gamma, beta) {
  const num  = 1 << n;
  const norm = 1 / Math.sqrt(num);
  const re   = new Float64Array(num).fill(norm);
  const im   = new Float64Array(num).fill(0);
  applyPhase(re, im, diag, gamma);
  applyMixer(re, im, n, beta);
  let cut = 0;
  for (let s = 0; s < num; s++) cut += (re[s] ** 2 + im[s] ** 2) * diag[s];
  return cut;
}

function qaojsExpCut(n, edges, gamma, beta) {
  const diag = computeCutDiag(n, edges);
  return qaojsExpCutFromDiag(n, diag, gamma, beta);
}

function wrapAngle(value, period) {
  const wrapped = value % period;
  return wrapped < 0 ? wrapped + period : wrapped;
}

function exactMaxCutFromDiag(diag) {
  let best = -Infinity;
  for (let index = 0; index < diag.length; index++) best = Math.max(best, diag[index]);
  return best;
}

function qaojsOptimize(n, edges) {
  const diag = computeCutDiag(n, edges);
  const evaluate = (gamma, beta) => {
    const wrappedGamma = wrapAngle(gamma, Math.PI);
    const wrappedBeta = wrapAngle(beta, Math.PI / 2);
    return {
      gamma: wrappedGamma,
      beta: wrappedBeta,
      cut: qaojsExpCutFromDiag(n, diag, wrappedGamma, wrappedBeta)
    };
  };

  const coarseGammaSteps = 16;
  const coarseBetaSteps = 12;
  let best = evaluate(Math.PI / 4, Math.PI / 8);

  for (let gi = 0; gi < coarseGammaSteps; gi++) {
    for (let bi = 0; bi < coarseBetaSteps; bi++) {
      const candidate = evaluate(
        (gi + 0.5) * Math.PI / coarseGammaSteps,
        (bi + 0.5) * Math.PI / (2 * coarseBetaSteps)
      );
      if (candidate.cut > best.cut) best = candidate;
    }
  }

  const seedPoints = [
    best,
    evaluate(Math.PI / 8, Math.PI / 10),
    evaluate(Math.PI / 3, Math.PI / 6),
    evaluate((2 * Math.PI) / 3, Math.PI / 5),
    evaluate(Math.random() * Math.PI, Math.random() * Math.PI / 2)
  ];

  function refineFrom(start) {
    let current = start;
    let stepGamma = Math.PI / 6;
    let stepBeta = Math.PI / 10;

    while (stepGamma > 1e-3 || stepBeta > 1e-3) {
      let improved = false;
      const gammaMoves = [0, -stepGamma, stepGamma];
      const betaMoves = [0, -stepBeta, stepBeta];

      for (const dGamma of gammaMoves) {
        for (const dBeta of betaMoves) {
          if (dGamma === 0 && dBeta === 0) continue;
          const candidate = evaluate(current.gamma + dGamma, current.beta + dBeta);
          if (candidate.cut > current.cut + 1e-9) {
            current = candidate;
            improved = true;
          }
        }
      }

      if (!improved) {
        stepGamma *= 0.5;
        stepBeta *= 0.5;
      }
    }

    return current;
  }

  for (const seed of seedPoints) {
    const refined = refineFrom(seed);
    if (refined.cut > best.cut) best = refined;
  }

  return {
    gammas: [best.gamma],
    betas: [best.beta],
    expected_cut: best.cut,
    exact_maxcut: exactMaxCutFromDiag(diag),
    solver: 'browser-classical-refinement'
  };
}

async function fetchPredictionViaApi(payload) {
  for (const baseUrl of API_BASE_CANDIDATES) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2500);

    try {
      const res = await fetch(`${baseUrl}/predict`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      if (!res.ok) continue;

      const data = await res.json();
      return { data, baseUrl };
    } catch (_) {
      clearTimeout(timeoutId);
    }
  }

  return null;
}

function formatApiLabel(baseUrl) {
  return baseUrl.replace(/^https?:\/\//, '');
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
    line.setAttribute('stroke', '#c4d0df');
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
    circle.setAttribute('fill', '#0b1f3a');
    circle.setAttribute('stroke', '#2f5daa');
    circle.setAttribute('stroke-width', '1.5');
    g.appendChild(circle);

    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttribute('x', p.x);
    text.setAttribute('y', p.y + 5);
    text.setAttribute('text-anchor', 'middle');
    text.setAttribute('fill', '#ffffff');
    text.setAttribute('font-size', '12');
    text.setAttribute('font-family', 'Source Sans 3, sans-serif');
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
  let resolvedApiBase = null;

  const apiResult = await fetchPredictionViaApi({n: currentN, edges: currentEdges});
  if (apiResult) {
    data = apiResult.data;
    resolvedApiBase = apiResult.baseUrl;
  } else {
    data   = qaojsOptimize(currentN, currentEdges);
    source = 'browser';
  }

  resGammas.textContent = data.gammas.map(v => v.toFixed(4)).join(', ');
  resBetas.textContent  = data.betas.map(v  => v.toFixed(4)).join(', ');
  resCut.textContent    = data.expected_cut.toFixed(4);

  const badge = document.getElementById('source-badge');
  if (source === 'browser') {
    badge.textContent = '⚡ computed in browser (exact p=1 simulation + local refinement)';
    badge.className = 'source-badge browser';
  } else {
    badge.textContent = `🖥 computed by GNN API (${formatApiLabel(resolvedApiBase)})`;
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
    line.setAttribute('stroke', isCut ? '#2f5daa' : '#c4d0df');
    line.setAttribute('stroke-width', isCut ? '2' : '1.5');
  });
}

// Initial render
renderGraph(currentN, currentEdges);
