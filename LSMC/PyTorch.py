import numpy as np, torch, time, sys
torch.set_default_tensor_type("torch.cuda.FloatTensor")

Spot = torch.tensor(36., requires_grad=True)
σ = torch.tensor(0.2, requires_grad=True)
K = torch.tensor(40., requires_grad=True)
r = torch.tensor(0.06, requires_grad=True)
n = 100000
m = 10
T = 1
order = int(sys.argv[1])
Δt = T / m


def chebyshev_basis(x, k):
    B = [torch.ones_like(x)]
    B.append(x)
    for n in range(2, k):
        B.append(2 * x * B[n - 1] - B[n - 2])

    return torch.stack(B, dim=1)


def ridge_regression(X, Y, λ=100):
    I = torch.eye(order)
    YY = Y.reshape(-1, 1)
    β = torch.gesv(X.transpose(1, 0) @ YY, X.transpose(1, 0) @ X + λ * I)[0]
    return torch.squeeze(X @ β)


def first_one(x):
    original = x
    x = (x > 0).type(x.dtype)
    nt = x.shape[0]
    batch_size = x.shape[1]
    x_not = 1 - x
    sum_x = torch.clamp(torch.cumprod(x_not, dim=0), max=1.)
    ones = torch.ones((1, batch_size))
    lag = sum_x[:(nt - 1), :]
    lag = torch.cat([ones, lag], dim=0)
    return original * (lag * x)


def scale(x):
    xmin = torch.min(x)
    xmax = torch.max(x)
    a = 2 / (xmax - xmin)
    b = -0.5 * a * (xmin + xmax)
    return a * x + b


def advance(S):
    dB = (np.sqrt(Δt) * torch.randn(n))
    out = S + r * S * Δt + σ * S * dB
    return out


def where(cond, x_1, x_2):
    cond = cond.type(x_1.dtype)
    return (cond * x_1) + ((1 - cond) * x_2)


def main():
    S = [Spot * torch.ones([n])]

    for t in range(m):
        S.append(advance(S[t]))

    S = torch.stack(S)

    discount = torch.exp(-r * Δt)
    CFL = torch.clamp(K - S, min=0.)

    value = [1] * m
    value[-1] = CFL[-1] * discount
    CV = [torch.zeros_like(S[0])] * m

    for k in range(1, m):
        t = m - k - 1
        t_next = t + 1

        X = chebyshev_basis(scale(S[t_next]), order)
        Y = value[t_next]
        CV[t] = ridge_regression(X, Y)
        value[t] = discount * where(CFL[t_next] > CV[t],
                                    CFL[t_next],
                                    value[t_next])

    CV = torch.stack(CV)
    POF = where(CV < CFL[1:], CFL[1:], torch.zeros_like(CV))
    FPOF = first_one(POF)
    m_range = torch.tensor(range(m), dtype=torch.float).reshape(-1, 1)
    dFPOF = FPOF * torch.exp(-r * m_range * Δt)
    price = torch.sum(dFPOF) / n
    price.backward()
    greeks = [Spot.grad, σ.grad, K.grad, r.grad]
    return [price] + greeks


result = main()

t0 = time.time()
for idx in range(100):
    main()
t1 = time.time()
print((t1 - t0) / 100 * 1000)
