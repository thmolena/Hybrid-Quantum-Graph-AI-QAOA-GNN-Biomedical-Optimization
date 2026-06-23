from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
from src.gnn import SimpleGCN
from src.data import graph_to_adj_feat
from src.qaoa_sim import qaoa_state, expected_cut

app = Flask(__name__)
CORS(app)

model = SimpleGCN(in_feats=1, hidden=32, out_feats=2, p=1)
try:
    model.load_state_dict(torch.load('model.pt'))
    model.eval()
except Exception:
    pass

@app.route('/predict', methods=['POST'])
def predict():
    body = request.get_json() or {}
    edges = body.get('edges', [])
    n = body.get('n', None)
    if n is None:
        # infer n
        nodes = set()
        for u,v in edges:
            nodes.add(u); nodes.add(v)
        n = max(nodes)+1
    # build graph adjacency
    A = np.zeros((n,n))
    for u,v in edges:
        A[u,v] = 1
        A[v,u] = 1
    A = A + np.eye(n)
    deg = A.sum(axis=1)
    X = deg.reshape(n,1)

    with torch.no_grad():
        At = torch.tensor(A, dtype=torch.float32)
        Xt = torch.tensor(X, dtype=torch.float32)
        out = model(Xt, At)  # [2, p]
        out = out.view(-1).numpy()
        p = out.shape[0]//2
        gammas = out[:p]
        betas = out[p:]

    state = qaoa_state(n, edges, gammas, betas)
    val = expected_cut(n, edges, state)
    return jsonify({'gammas': gammas.tolist(), 'betas': betas.tolist(), 'expected_cut': float(val)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
