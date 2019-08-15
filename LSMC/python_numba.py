import numpy as np
import time
from numba import autojit

Spot = 36
σ = 0.2
K = 40
r = 0.06
n = 100000
m = 10
T = 1
order = 25
Δt = T / m


@autojit
def chebyshev_basis(x, k):
    B = np.ones((k, len(x)))
    B[1, :] = x
    for n in range(2, k):
        B[n, :] = 2 * x * B[n - 1, :] - B[n - 2, :]

    return B.T


@autojit
def ridge_regression(X, Y, λ=100):
    I = np.eye(order)
    β = np.linalg.solve(X.T @ X + λ * I, X.T @ Y)
    return X @ β


@autojit
def first_one(x):
    original = x
    x = np.greater(x, 0.)
    nt = x.shape[0]
    batch_size = x.shape[1]
    x_not = 1 - x
    sum_x = np.minimum(np.cumprod(x_not, axis=0), 1.)
    ones = np.ones((1, batch_size,))
    lag = sum_x[:(nt - 1), :]
    lag = np.vstack([ones, lag])
    return original * (lag * x)


@autojit
def scale(x):
        xmin = x.min()
        xmax = x.max()
        a = 2 / (xmax - xmin)
        b = 1 - a * xmax
        return a * x + b


@autojit
def advance(S, r, σ, Δt, n):
    dB = np.sqrt(Δt) * np.random.normal(size=[n])
    out = S + r * S * Δt + σ * S * dB
    return out


@autojit
def compute_price(order, Spot, σ, K, r):
    np.random.seed(0)
    S = np.zeros((m + 1, n))
    S[0, :] = Spot

    for t in range(m):
        S[t + 1, :] = advance(S[t, :], r, σ, Δt, n)

    discount = np.exp(-r * Δt)
    CFL = np.maximum(0., K - S)
    value = np.zeros((m, n))
    value[-1] = CFL[-1] * discount
    CV = np.zeros((m, n))

    for k in range(2, m + 1):
        t = m - k
        t_next = t + 1

        X = chebyshev_basis(scale(S[t_next]), order)
        Y = value[t_next]
        CV[t] = ridge_regression(X, Y)
        value[t] = discount * np.where(CFL[t_next] > CV[t],
                                       CFL[t_next],
                                       value[t_next])

    POF = np.where(CV < CFL[1:], CFL[1:], 0)
    FPOF = first_one(POF)
    m_range = np.array(range(m)).reshape(-1, 1)
    dFPOF = FPOF * np.exp(-r * m_range * Δt)
    PRICE = dFPOF.sum() / n
    return PRICE


compute_price(order, Spot, σ, K, r)  # warmup
ε = 1e-2
t0 = time.time()
P = compute_price(order, Spot, σ, K, r)
dP_dS = (compute_price(order, Spot + ε, σ, K, r) - P) / ε
dP_dσ = (compute_price(order, Spot, σ + ε, K, r) - P) / ε
dP_dK = (compute_price(order, Spot, σ, K + ε, r) - P) / ε
dP_dr = (compute_price(order, Spot, σ, K, r + ε) - P) / ε
t1 = time.time()
print((t1 - t0) * 1000)  # Multiply by four bc we need the greeks
