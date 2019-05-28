import time, numpy as np, tensorflow as tf


logy_grid = np.loadtxt('logy_grid.txt', dtype=np.float32)
Py = np.loadtxt('P.txt', dtype=np.float32)


def main(nB=351, repeats=500):
    β = .953
    γ = 2.
    r = 0.017
    θ = 0.282
    ny = len(logy_grid)

    Bgrid = tf.linspace(-.45, .45, nB)
    ygrid = tf.exp(logy_grid)

    Vd = tf.Variable(tf.zeros([ny, 1]))
    Vc = tf.Variable(tf.zeros((ny, nB)))
    V = tf.Variable(tf.zeros((ny, nB)))
    Q = tf.Variable(tf.ones((ny, nB)) * .95)

    # Expectations
    EV = Py @ V
    EVd = Py @ Vd
    EVc = Py @ Vc

    y = tf.reshape(ygrid, [-1, 1, 1])
    B = tf.reshape(Bgrid, [1, -1, 1])
    Bnext = tf.reshape(Bgrid, [1, 1, -1])

    zero_ind = nB // 2

    # Utility function
    def u(c):
        return c**(1 - γ) / (1 - γ)

    ymean = tf.reduce_mean(ygrid)
    def_y = tf.minimum(0.969 * ymean, ygrid)

    Vd_target = u(def_y) + β * (θ * EVc[:, zero_ind] + (1 - θ) * EVd[:, 0])
    Vd_target = tf.reshape(Vd_target, [-1, 1])

    Qnext = tf.expand_dims(Q, axis=1)

    c = tf.maximum(y - Qnext * Bnext + B, 1e-14)
    EV = tf.expand_dims(EV, axis=1)
    m = u(c) + β * EV
    Vc_target = tf.reduce_max(m, axis=2)

    # Update prices
    default_states = tf.cast(Vd > Vc, tf.float32)
    default_prob = Py @ default_states
    Q_target = (1 - default_prob) / (1 + r)

    # Value function
    V_upd = tf.maximum(Vc, Vd)

    update_V = V.assign(V_upd)
    update_Vc = Vc.assign(Vc_target)
    update_Vd = Vd.assign(Vd_target)

    with tf.control_dependencies([update_Vc]):
        with tf.control_dependencies([update_Vd]):
            with tf.control_dependencies([update_V]):
                update = Q.assign(Q_target)

    # Execution -----------------------------------
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = 1
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    sess.run(update)
    t0 = time.time()
    for counter in range(repeats):
        sess.run(update)
    t1 = time.time()
    out = (t1 - t0) / repeats

    return out


print(1000 * main(nB=1551, repeats=50))
