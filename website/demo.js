const graphDiv = document.getElementById('graph');
const nInput = document.getElementById('n');
const API_BASE_URL = (window.API_BASE_URL || 'http://localhost:5000').replace(/\/$/, '');
let currentEdges = [];

function renderGraph(n){
  graphDiv.innerHTML = '';
  for(let i=0;i<n;i++){
    const el = document.createElement('div');
    el.className = 'node';
    el.textContent = i;
    graphDiv.appendChild(el);
  }
}

document.getElementById('random').addEventListener('click', ()=>{
  const n = parseInt(nInput.value);
  currentEdges = [];
  for(let i=0;i<n;i++){
    for(let j=i+1;j<n;j++){
      if(Math.random()<0.4) currentEdges.push([i,j]);
    }
  }
  renderGraph(n);
  document.getElementById('result').textContent = 'Random graph generated with '+currentEdges.length+' edges';
});

document.getElementById('predict').addEventListener('click', async ()=>{
  const n = parseInt(nInput.value);
  const resultEl = document.getElementById('result');
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({n: n, edges: currentEdges})
    });
    if (!res.ok) {
      throw new Error(`Request failed (${res.status})`);
    }
    const data = await res.json();
    resultEl.textContent = 'Gammas: '+JSON.stringify(data.gammas)+" Betas: "+JSON.stringify(data.betas)+" Expected Cut: "+data.expected_cut;
  } catch (err) {
    resultEl.textContent = `Prediction failed. Ensure the API is running at ${API_BASE_URL}. ${err.message}`;
  }
});

renderGraph(parseInt(nInput.value));
