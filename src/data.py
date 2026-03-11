import networkx as nx
import numpy as np


def sample_erdos_renyi(n, p_edge=0.5, seed=None):
    G = nx.erdos_renyi_graph(n, p_edge, seed=seed)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, p_edge, seed=seed)
    return G


def graph_to_adj_feat(G):
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G)
    # add self loops
    A = A + np.eye(n)
    deg = A.sum(axis=1)
    X = np.expand_dims(deg, axis=1)  # simple node feature: degree
    return A, X
