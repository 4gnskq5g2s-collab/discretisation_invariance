import jax.numpy as jnp

from jax import jit, random
from jax.lax import scan

def get_coordinates(N):
    x = jnp.linspace(0, 1, N+2)[1:-1]
    x_extended = jnp.linspace(0, 1, 2*N + 3)[1:-1]
    return x, x_extended

def get_A(a_extended):
    A = jnp.diag(-a_extended[::2][:-1]-a_extended[::2][1:]) + jnp.diag(a_extended[::2][1:-1], k=-1) + jnp.diag(a_extended[::2][1:-1], k=+1)
    return A

def get_B(N):
    B = jnp.diag(jnp.ones((N-1,))/2, k=+1) - jnp.diag(jnp.ones((N-1,))/2, k=-1)
    B = B.at[0, :3].set(jnp.array([-3/2, 2, -1/2]))
    B = B.at[-1, -3:].set(jnp.array([1/2, -2, 3/2]))
    return B

def integration_step_scan_(carry, t, N_steps):
    u, f, A, B = carry
    v = jnp.copy(u)
    for _ in range(N_steps):
        r = B @ v**2 / 2 + A @ v - u + f
        v = v - jnp.linalg.solve(B @ jnp.diag(v) + A, r)
    return [v, f, A, B], v

def integrate_Burgers(u, f, A, B, N_newton, t):
    integration_step = lambda carry, t: integration_step_scan_(carry, t, N_newton)
    U = scan(integration_step, [u, f, A, B], t)[1]
    return U

def get_weights(N, n, alpha):
    w = jnp.fft.rfftfreq(N)
    w = 1 / (1 + alpha*(2*jnp.pi*w)**2)**n
    return w

def get_random_function(key, w, N):
    c = random.normal(key, (w.shape[0],), dtype=jnp.complex64) * w
    f = jnp.fft.irfft(c, n=N)
    return f

def get_initial_conditions(key, w, N):
    f = get_random_function(key, w, N)
    return f * jnp.sqrt(N) * 5

def get_diffusion_coefficient(key, w, N):
    f = get_random_function(key, w, N)
    f = f * jnp.sqrt(N)
    f = 0.005 + (1 + jnp.tanh(30*f))/2 * 0.1
    return f

def get_dataset(key, N_samples):
    N_x, N_t = 512, 64
    T_max = 0.5
    N_newton = 3
    
    x, x_extended = get_coordinates(N_x)
    project = jnp.sin(jnp.pi*x)
    t = jnp.linspace(0, T_max, N_t)[1:]
    dt = t[1] - t[0]
    h = x[1] - x[0]
    B = dt * get_B(N_x) / h
    
    n = 2
    alpha = 50
    w_u = get_weights(N_x, n, alpha)
    a_extended = jnp.ones((2*N_x + 1, )) * 1e-1
    A = jnp.eye(N_x) - dt * get_A(a_extended) / h**2

    keys = random.split(key, (N_samples, ))
    data = {
        "init": [],
        "sol": [],
        "x": x
    }
    for key in keys:
        u = get_initial_conditions(key, w_u, N_x) * project
        U = integrate_Burgers(u, u*0, A, B, N_newton, t)
        data["init"].append(u)
        data["sol"].append(U[-1])
    for key in data.keys():
        data[key] = jnp.array(data[key])

    return data

if __name__ == "__main__":
    key = random.PRNGKey(67)
    N_samples = 1000
    data = get_dataset(key, N_samples)
    jnp.savez("Burgers_dataset.npz", **data)
