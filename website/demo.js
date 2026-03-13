const API_BASE_URL = (window.API_BASE_URL || 'http://localhost:5000').replace(/\/$/, '');

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
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({n: currentN, edges: currentEdges})
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const data = await res.json();

    resGammas.textContent = data.gammas.map(v => v.toFixed(4)).join(', ');
    resBetas.textContent  = data.betas.map(v => v.toFixed(4)).join(', ');
    resCut.textContent    = data.expected_cut.toFixed(4);
    resultPanel.classList.remove('hidden');

    // Colour edges by cut value (simple visual feedback)
    highlightCut(currentN, currentEdges, data.gammas, data.betas);
  } catch (err) {
    showError(`API unreachable at <code>${API_BASE_URL}</code>. Run <code>python -m src.server</code> locally.`);
  }
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
