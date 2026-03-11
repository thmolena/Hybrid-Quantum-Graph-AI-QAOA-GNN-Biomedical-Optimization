import numpy as np

def get_maxcut_operator(n, edges):
    # Returns the diagonal matrix with cut contributions per basis state
    num_states = 2**n
    diag = np.zeros(num_states)
    for s in range(num_states):
        bits = [(s>>i)&1 for i in range(n)]
        val = 0
        for (u,v) in edges:
            if bits[u] != bits[v]:
                val += 1
        diag[s] = val
    return np.diag(diag)

def apply_phase(state, n, edges, gamma):
    # diagonal unitary e^{-i gamma C}
    diag = np.exp(-1j * gamma * np.diag(get_maxcut_operator(n, edges)))
    return state * diag

def apply_mixer(state, n, beta):
    # mixer exp(-i beta B) where B = sum X_i
    # For small n, construct full unitary
    from scipy.linalg import expm
    X = np.array([[0,1],[1,0]])
    B = None
    for i in range(n):
        ops = [np.eye(2)]*n
        ops[i] = X
        term = ops[0]
        for op in ops[1:]:
            term = np.kron(term, op)
        if B is None:
            B = term
        else:
            B = B + term
    U = expm(-1j * beta * B)
    return U.dot(state)


def qaoa_state(n, edges, gammas, betas):
    # start in uniform superposition
    num_states = 2**n
    state = np.ones(num_states, dtype=complex)/np.sqrt(num_states)
    p = len(gammas)
    for i in range(p):
        state = apply_phase(state, n, edges, gammas[i])
        state = apply_mixer(state, n, betas[i])
    return state


def expected_cut(n, edges, state):
    C = get_maxcut_operator(n, edges)
    return np.real(np.conjugate(state).T.dot(C.dot(state)))
