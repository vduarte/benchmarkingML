import numpy as np, time, sys


Spot = 36
σ = 0.2
K = 40
r = 0.06
n = 100000
m = 10
T = 1
order = int(sys.argv[1])
Δt = T / m


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


def chebyshev_basis(x, k):
    B = np.ones((k, len(x)))
    B[1, :] = x
    for n in range(2, k):
        B[n, :] = 2 * x * B[n - 1, :] - B[n - 2, :]

    return B.T


def ridge_regression(X, Y, λ=100):
    I = np.eye(order)
    β = np.linalg.solve(X.T @ X + λ * I, X.T @ Y)
    return X @ β


def scale(x):
        xmin = x.min()
        xmax = x.max()
        a = 2 / (xmax - xmin)
        b = 1 - a * xmax
        return a * x + b


def advance(S):
    dB = np.sqrt(Δt) * np.random.normal(size=[n])
    out = S + r * S * Δt + σ * S * dB
    return out


def main():
    S = np.zeros((m + 1, n))
    S[0, :] = Spot

    for t in range(m):
        S[t + 1, :] = advance(S[t, :])

    discount = np.exp(-r * Δt)
    CFL = np.maximum(0., K - S)

    value = np.zeros((m, n))
    value[-1] = CFL[-1] * discount
    CV = np.zeros((m, n))

    for k in range(1, m):
        t = m - k - 1
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


t0 = time.time()
main()
t1 = time.time()
print((t1 - t0) * 4 * 1000)  # Multiply by four bc we need the greeks
