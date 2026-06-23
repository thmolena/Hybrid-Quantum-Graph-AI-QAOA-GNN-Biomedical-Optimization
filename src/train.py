import argparse
import torch
import torch.optim as optim
import numpy as np
from src.gnn import SimpleGCN
from src.data import sample_erdos_renyi, graph_to_adj_feat
from src.qaoa_sim import qaoa_state, expected_cut
from scipy.optimize import minimize


def optimize_angles(n, edges, p):
    # optimize expected_cut with classical minimizer (maximize -> minimize negative)
    def loss_fn(x):
        gammas = x[:p]
        betas = x[p:]
        state = qaoa_state(n, edges, gammas, betas)
        val = expected_cut(n, edges, state)
        return -val

    x0 = np.random.uniform(0, np.pi, size=2*p)
    res = minimize(loss_fn, x0, method='Nelder-Mead', options={'maxiter':200})
    sol = res.x
    return sol[:p], sol[p:]


def build_dataset(size, n, p):
    data = []
    for i in range(size):
        G = sample_erdos_renyi(n, p_edge=0.5, seed=i)
        edges = list(G.edges())
        A, X = graph_to_adj_feat(G)
        gammas, betas = optimize_angles(n, edges, p)
        target = np.concatenate([gammas, betas])
        data.append((A, X, target, edges))
    return data


def train(args):
    device = torch.device('cpu')
    p = args.p
    out_feats = 2  # predicts gamma and beta per layer as 2 scalars
    model = SimpleGCN(in_feats=1, hidden=32, out_feats=2, p=p)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    data = build_dataset(args.dataset_size, args.n, p)
    Xs = [torch.tensor(x[1], dtype=torch.float32) for x in data]
    As = [torch.tensor(x[0], dtype=torch.float32) for x in data]
    Ys = [torch.tensor(x[2], dtype=torch.float32) for x in data]

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        for A, X, Y in zip(As, Xs, Ys):
            pred = model(X, A)  # [2, p]
            pred = pred.view(-1)
            loss = ((pred - Y)**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} loss={total_loss/len(As):.6f}")

    torch.save(model.state_dict(), args.model_path)
    print('Saved model to', args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-size', type=int, default=20)
    parser.add_argument('--n', type=int, default=6)
    parser.add_argument('--p', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--model-path', type=str, default='model.pt')
    args = parser.parse_args()
    train(args)
