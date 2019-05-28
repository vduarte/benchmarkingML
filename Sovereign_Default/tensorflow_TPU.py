import time, numpy as np, tensorflow as tf


# Load grid for log(y) and transition matrix
logy_grid = np.loadtxt('logy_grid.txt').astype(np.float32)
Py = np.loadtxt('P.txt').astype(np.float32)


def main(nB=351, repeats=500):
    # Model parameters
    β = .953
    γ = 2.
    r = 0.017
    θ = 0.282

    # hyperparameters
    ny = len(logy_grid)

    # Grids
    Bgrid = tf.linspace(-.45, .45, nB)
    ygrid = tf.exp(logy_grid)

    # Initialize tensors
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

    # Compute Vd
    Vd_target = u(def_y) + β *\
        (θ * EVc[:, zero_ind] + (1 - θ) * EVd[:, 0])
    Vd_target = tf.reshape(Vd_target, [-1, 1])

    Qnext = tf.expand_dims(Q, axis=1)
    EV = tf.expand_dims(EV, axis=1)

    def compute_Vtarget(y, Qnext, EV):
        c = tf.maximum(y - Qnext * Bnext + B, 1e-14)
        m = u(c) + β * EV
        Vc_target = tf.reduce_max(m, axis=2)
        return Vc_target

    Vc_target = tf.contrib.tpu.shard(
        compute_Vtarget, inputs=[y, Qnext, EV], num_shards=8)[0]

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
    address = tf.contrib.cluster_resolver.TPUClusterResolver().get_master()
    sess = tf.Session(address)
    sess.run(tf.contrib.tpu.initialize_system())
    sess.run(tf.global_variables_initializer())
    sess.run(update)  # warm up

    t0 = time.time()
    for counter in range(repeats):
        sess.run(update)
    t1 = time.time()
    out = (t1 - t0) / repeats
    sess.close()
    return out


print(1000 * main(nB=951, repeats=500))
