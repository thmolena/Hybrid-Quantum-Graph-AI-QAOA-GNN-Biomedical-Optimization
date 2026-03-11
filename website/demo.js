const graphDiv = document.getElementById('graph');
const nInput = document.getElementById('n');
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
  const res = await fetch('/predict', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({n: n, edges: currentEdges})
  });
  const data = await res.json();
  document.getElementById('result').textContent = 'Gammas: '+JSON.stringify(data.gammas)+" Betas: "+JSON.stringify(data.betas)+" Expected Cut: "+data.expected_cut;
});

renderGraph(parseInt(nInput.value));
