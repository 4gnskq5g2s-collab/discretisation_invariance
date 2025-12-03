import jax.numpy as jnp

from jax import jit, random
from jax.lax import scan

def get_weights(N, h, decay, power):
    k = 2*jnp.pi*jnp.fft.fftfreq(N, d=h)
    w = 1 / (1 + decay*k**2)**power
    return w, k

def random_series(w, k, c_a):
    res = jnp.real(jnp.fft.ifftn(c_a * w, norm="ortho"))
    return res

def a_rand(N, params, key):
    w, k, alpha, beta, slope, A = params
    c_a = random.normal(key, (N,), dtype=jnp.complex64) * A
    s = random_series(w, k, c_a)
    s = s - jnp.mean(s)
    a = alpha + (beta - alpha)*(jnp.tanh(slope*s) + 1)/2
    return a

def rhs_rand(N, params, key):
    w, k, A, shift = params
    c_rhs = random.normal(key, (N,), dtype=jnp.complex64) * A
    rhs = random_series(w, k, c_rhs) + shift
    return rhs

def get_dataset(key, N_samples, use_b=True):
    N = 512
    decay_a = 1e-3
    power_a = 4
    alpha_a = 0.1
    beta_a = 81.0
    slope_a = 50
    A_a = 40.0
    decay_b = 1e-3
    power_b = 2
    A_b = 20.0
    shift_b = 0.0

    x = jnp.linspace(0, 1, N+2)[1:-1]
    M = 2*N + 1
    h = x[1] - x[0]

    w_a, k_a = get_weights(M, h/2, decay_a, power_a)
    params_a = [w_a, k_a, alpha_a, beta_a, slope_a, A_a]

    w_b, k_b = get_weights(N, (x[1]-x[0]), decay_b, power_b)
    params_b = [w_b, k_b, A_b, shift_b]

    Keys = random.split(key, (N_samples, 2))
    data = {
        "a": [],
        "b": [],
        "s": [],
        "x": x
    }
    
    for keys in Keys:
        a = a_rand(M, params_a, keys[0])
        if use_b:
            b = rhs_rand(N, params_b, keys[1])
        else:
            b = jnp.ones_like(x)
        A = (jnp.diag(a[::2][:-1] + a[::2][1:], k=0) - jnp.diag(a[::2][1:-1], k=1) - jnp.diag(a[::2][1:-1], k=-1))
        s = jnp.linalg.solve(A, b*h**2)
        data["a"].append(a[1::2])
        if use_b:
            data["b"].append(b)
        data["s"].append(s)

    for key in ["a", "b", "s"]:
        data[key] = jnp.array(data[key])
    if not use_b:
        del data['b']
    return data

if __name__ == "__main__":
    key = random.PRNGKey(71)
    N_samples = 1000
    data = get_dataset(key, N_samples, use_b=False)
    jnp.savez("diffusion_dataset.npz", **data)
