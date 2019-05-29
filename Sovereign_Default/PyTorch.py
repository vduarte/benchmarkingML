import torch, numpy as np, time
torch.set_default_tensor_type("torch.cuda.FloatTensor")

logy_grid = torch.tensor(np.loadtxt('logy_grid.txt', dtype=np.float32))
Py = torch.tensor(np.loadtxt('P.txt', dtype=np.float32))


def main(nB=151, repeats=500):
    β = .953
    γ = 2.
    r = 0.017
    θ = 0.282
    ny = len(logy_grid)

    Bgrid = torch.linspace(-.45, .45, nB)
    ygrid = torch.exp(logy_grid)

    ymean = torch.mean(ygrid)
    def_y = torch.min(0.969 * ymean, ygrid)

    Vd = torch.zeros([ny, 1])
    Vc = torch.zeros((ny, nB))
    V = torch.zeros((ny, nB))
    Q = torch.ones((ny, nB)) * .95

    y = torch.reshape(ygrid, [-1, 1, 1])
    B = torch.reshape(Bgrid, [1, -1, 1])
    Bnext = torch.reshape(Bgrid, [1, 1, -1])

    zero_ind = nB // 2

    def u(c):
        return c**(1 - γ) / (1 - γ)

    def iterate(V, Vc, Vd, Q):
        EV = torch.matmul(Py, V)
        EVd = torch.matmul(Py, Vd)
        EVc = torch.matmul(Py, Vc)

        # Compute Vd
        Vd_target = u(def_y) + β * (θ * EVc[:, zero_ind] + (1 - θ) * EVd[:, 0])
        Vd_target = torch.reshape(Vd_target, [-1, 1])

        Qnext = torch.reshape(Q, [ny, 1, nB])

        c = torch.relu(y - Qnext * Bnext + B)
        EV = torch.reshape(EV, [ny, 1, nB])
        m = u(c) + β * EV
        Vc_target = torch.max(m, dim=2, out=None)[0]

        default_states = (Vd > Vc).float()
        default_prob = torch.matmul(Py, default_states)
        Q_target = (1 - default_prob) / (1 + r)

        V_target = torch.max(Vc, Vd)

        return V_target, Vc_target, Vd_target, Q_target

    iterate = torch.jit.trace(iterate, (V, Vc, Vd, Q))  # Jit compilation
    t0 = time.time()
    for iteration in range(repeats):
        V, Vc, Vd, Q = iterate(V, Vc, Vd, Q)
    t1 = time.time()

    return (t1 - t0) / repeats


main(nB=151, repeats=50)  # warmup
print(1000 * main(nB=1351, repeats=500))
