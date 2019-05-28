import numpy as np, time

logy_grid = np.loadtxt('logy_grid.txt')
Py = np.loadtxt('P.txt')


def main(nB=351, repeats=500):
    β = .953
    γ = 2.
    r = 0.017
    θ = 0.282
    ny = len(logy_grid)

    Bgrid = np.linspace(-.45, .45, nB)
    ygrid = np.exp(logy_grid)

    ymean = np.mean(ygrid)
    def_y = np.minimum(0.969 * ymean, ygrid)

    Vd = np.zeros([ny, 1])
    Vc = np.zeros((ny, nB))
    V = np.zeros((ny, nB))
    Q = np.ones((ny, nB)) * .95

    y = np.reshape(ygrid, [-1, 1, 1])
    B = np.reshape(Bgrid, [1, -1, 1])
    Bnext = np.reshape(Bgrid, [1, 1, -1])

    zero_ind = nB // 2

    def u(c):
        return c**(1 - γ) / (1 - γ)

    def iterate(V, Vc, Vd, Q):
        EV = np.dot(Py, V)
        EVd = np.dot(Py, Vd)
        EVc = np.dot(Py, Vc)

        Vd_target = u(def_y) + β * (θ * EVc[:, zero_ind] + (1 - θ) * EVd[:, 0])
        Vd_target = np.reshape(Vd_target, [-1, 1])

        Qnext = np.reshape(Q, [ny, 1, nB])

        c = np.maximum(y - Qnext * Bnext + B, 1e-14)
        EV = np.expand_dims(EV, axis=1)
        m = u(c) + β * EV
        Vc_target = np.max(m, axis=2)

        default_states = Vd > Vc
        default_prob = np.dot(Py, default_states)
        Q_target = (1 - default_prob) / (1 + r)

        V_target = np.maximum(Vc, Vd)

        return V_target, Vc_target, Vd_target, Q_target

    iterate(V, Vc, Vd, Q)  # warmup
    t0 = time.time()
    for iteration in range(repeats):
        V, Vc, Vd, Q = iterate(V, Vc, Vd, Q)
    t1 = time.time()

    return (t1 - t0) / repeats


print(1000 * main(nB=1151, repeats=100))

# _,V_ = main(nB=151, repeats=100)
