import numpy as np, tensorflow as tf, time

Spot = tf.Variable(36.)
σ = tf.Variable(0.2)
n = 12500
m = 10
K = tf.Variable(40.)
r = tf.Variable(0.06)
T = 1
order = 25
Δt = T / m


def chebyshev_basis(x, k):
    B = [tf.ones_like(x)]
    B.append(x)
    for n in range(2, k):
        B.append(2 * x * B[n - 1] - B[n - 2])

    return tf.stack(B, axis=1)


def ridge_regression(X, Y, λ=100):
    β = tf.linalg.lstsq(X, tf.reshape(Y, [-1, 1]), l2_regularizer=100)
    return tf.squeeze(X @ β)


def first_one(x):
    original = x
    x = tf.cast(tf.greater(x, 0.), dtype=x.dtype)
    nt = x.shape.as_list()[0]
    batch_size = x.shape.as_list()[1]
    x_not = 1 - x
    sum_x = tf.minimum(tf.cumprod(x_not, axis=0), 1.)
    ones = tf.ones([1, batch_size])
    lag = sum_x[:(nt - 1), :]
    lag = tf.concat([ones, lag], axis=0)
    return original * (lag * x)


def scale(x):
    xmin = tf.reduce_min(x)
    xmax = tf.reduce_max(x)
    a = 2 / (xmax - xmin)
    b = 1 - a * xmax
    return a * x + b


def advance(S):
    dB = np.sqrt(Δt) * tf.random_normal(shape=[n])
    out = S + r * S * Δt + σ * S * dB
    return out


def main():
    S = [Spot * tf.ones([n])]

    for t in range(m):
        S.append(advance(S[t]))

    S = tf.stack(S)

    discount = tf.exp(-r * Δt)
    CFL = tf.maximum(0., K - S)

    value = [1] * m
    value[-1] = CFL[-1] * discount
    CV = [tf.zeros_like(S[0])] * m

    for k in range(1, m):
        t = m - k - 1
        t_next = t + 1

        X = chebyshev_basis(scale(S[t_next]), order)
        Y = value[t_next]
        CV[t] = ridge_regression(X, Y)
        value[t] = discount * tf.where(CFL[t_next] > CV[t],
                                       CFL[t_next],
                                       value[t_next])

    CV = tf.stack(CV)
    POF = tf.where(CV < CFL[1:], CFL[1:], tf.zeros_like(CV))
    FPOF = first_one(POF)
    m_range = np.array(range(m)).reshape(-1, 1)
    dFPOF = FPOF * tf.exp(-r * m_range * Δt)
    price = tf.reduce_sum(dFPOF) / n
    greeks = tf.gradients(price, [Spot, σ, K, r])
    return price


results = tf.reduce_mean(tf.contrib.tpu.batch_parallel(main, num_shards=8)[0])

# ==============================================================================
# %% Run
# ==============================================================================
address = tf.contrib.cluster_resolver.TPUClusterResolver().get_master()
sess = tf.Session(address)
sess.run(tf.contrib.tpu.initialize_system())
sess.run(tf.global_variables_initializer())

sess.run(results)
t0 = time.time()
for idx in range(100):
    sess.run(results)
t1 = time.time()
print((t1 - t0) / 100 * 1000)
sess.close()
