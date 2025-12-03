import sys
import os
import time
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix
from jax import random, vmap
from jax.lax import scan, dot_general, dynamic_slice_in_dim
from jax.tree_util import tree_map, tree_flatten
from jax.nn import gelu
from jax.scipy.fft import dct

class DeepONet(eqx.Module):
    trunk_W: list
    trunk_b: list
    branch_W: list
    branch_b: list

    def __init__(self, N_layers, N_f_branch, D, key, s=1.0):
        n_obs, n_basis = N_f_branch
        N_branch, N_trunk = N_layers

        keys = random.split(key)
        keys_branch = random.split(keys[0], N_layers[0])
        keys_trunk = random.split(keys[1], N_layers[1])
        n_branch = [n_obs, ] + [n_basis, ]*N_branch
        n_trunk = [D, ] + [n_basis, ]*N_trunk

        self.trunk_W = [s*random.normal(key, (n_out, n_in)) for key, n_out, n_in in zip(keys_trunk, n_trunk[1:], n_trunk[:-1])]
        self.branch_W = [s*random.normal(key, (n_out, n_in)) for key, n_out, n_in in zip(keys_branch, n_branch[1:], n_branch[:-1])]
        self.trunk_b = [jnp.zeros([n_basis,] + [1,]*D) for _ in range(N_layers[1])]
        self.branch_b = [jnp.zeros((n_basis,)) for _ in range(N_layers[0])]

    def __call__(self, feature, x, basis, inv_G):
        c = self.branch_net(feature, x, basis, inv_G)
        phi = self.trunk_net(x)
        res = jnp.expand_dims(c @ phi, 0)
        return res

    def branch_net(self, feature, x, basis, inv_G):
        u = vmap(project, in_axes=(0, None, None))(feature, basis, inv_G).reshape(-1,)
        for W, b in zip(self.branch_W, self.branch_b):
            u = gelu(W @ u + b)
        return u

    def trunk_net(self, x):
        s = 10
        for W, b in zip(self.trunk_W, self.trunk_b):
            x = jnp.sin(s * (W @ x + b))
            s = 1
        return x

def get_observation_data(N_basis, x):
    x_ = jnp.linspace(0, 1, N_basis+2)
    I = jnp.eye(N_basis)
    b = jnp.array([jnp.interp(x, x_, jnp.pad(e, (1, 1))) for e in I])
    inv_G = jnp.linalg.inv(b @ b.T)
    return b, inv_G

def project(f, b, inv_G):
    return inv_G @ (f @ b.T)

def l2_loss(model, input, target, x, b, inv_G):
    X = model(input, x, b, inv_G)
    error = jnp.mean(jnp.sum(((X - target).reshape(target.shape[0], -1,))**2, axis=1))
    return error

def batch_l2_loss(model, input, target, x, b, inv_G):
    res = vmap(l2_loss, in_axes=(None, 0, 0, None, None, None))(model, input, target, x, b, inv_G)
    return jnp.mean(res)

l2_compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)

def l2_make_step_scan(carry, n, optim):
    model, features, targets, x, b, inv_G, opt_state = carry
    loss, grads = l2_compute_loss_and_grads(model, features[n], targets[n], x, b, inv_G)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, b, inv_G, opt_state], loss

def make_prediction_scan(carry, i):
    model, features, coords, b, inv_G = carry
    prediction = model(features[i], coords, b, inv_G)
    return carry, prediction

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    key = random.PRNGKey(679)
    if dataset_name == 'Burgers':
        data = jnp.load("Burgers_dataset.npz")
        for key in data.keys():
            print(key, data[key].shape)

        features = jnp.expand_dims(data["init"], 1)
        targets = jnp.expand_dims(data["sol"], 1)
        coordinates = jnp.expand_dims(data["x"], 0)

        features = features / jnp.max(jnp.abs(features))
        targets = targets / jnp.max(jnp.abs(targets))
    else:
        data = jnp.load("diffusion_dataset.npz")
        for key in data.keys():
            print(key, data[key].shape)

        features = jnp.expand_dims(data["a"], 1)
        targets = jnp.expand_dims(data["s"], 1)
        coordinates = jnp.expand_dims(data["x"], 0)

        features = features / jnp.max(jnp.abs(features))
        targets = targets / jnp.max(jnp.abs(targets))

    Data = "N_train,N_inference,train_error,test_error"
    for J in [3, 2, 1, 0]:
        N_layers = [6, 6]
        n_obs = 60
        n_basis = 100
        N_epoch = 1000
        N_drop = 100
        N_train = 900
        N_batch = 11
        D = features.ndim - 2
        learning_rate = 1e-3
        gamma = 0.5
        key = random.PRNGKey(45)
        keys = random.split(key)

        N_run = N_epoch * N_train // N_batch
        N_drop = N_drop * N_train // N_batch

        N_f_branch = [n_obs, n_basis]
        model = DeepONet(N_layers, N_f_branch, D, keys[0], s=1e-2)
        b, inv_G = get_observation_data(n_obs, coordinates[0, ::2**J])
        model_size = sum(tree_map(lambda x: jnp.size(x) if x.dtype == jnp.float32 else 2*jnp.size(x), tree_flatten(model)[0], is_leaf=eqx.is_array))
        #learning_rate_ = optax.exponential_decay(learning_rate, N_drop, gamma)
        optim = optax.adamw(learning_rate=learning_rate, weight_decay=0.0)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        n = random.choice(keys[1], N_train, shape = (N_run, N_batch))
        carry = [model, features[:, :, ::2**J], targets[:, :, ::2**J], coordinates[:, ::2**J], b, inv_G, opt_state]

        make_step_scan_ = lambda a, b: l2_make_step_scan(a, b, optim)

        start = time.time()
        carry, history = scan(make_step_scan_, carry, n)
        stop = time.time()
        training_time = stop - start
        model = carry[0]
        opt_state = carry[-1]

        print(f"training time {stop - start}")

        ind = jnp.arange(features.shape[0])
        J_ = J - jnp.arange(J+1)
        N_x_train = coordinates[:, ::2**J].shape[1]

        for j in J_:
            N_x = coordinates[:, ::2**j].shape[1]
            b, inv_G = get_observation_data(n_obs, coordinates[0, ::2**j])
            predictions = scan(make_prediction_scan, [model, features[:, :, ::2**j], coordinates[:, ::2**j], b, inv_G], ind)[1]
            errors = jnp.linalg.norm(predictions - targets[:, :, ::2**j], axis=2) / jnp.linalg.norm(targets[:, :, ::2**j], axis=2)
            train_error = jnp.mean(errors[:N_train])
            test_error = jnp.mean(errors[N_train:])
            print(f"N_x_train = {N_x_train}, N_x_inference = {N_x}, train error {train_error}, test error {test_error}")
            Data += f"\n{N_x_train},{N_x},{train_error},{test_error}"

    with open(f"DeepONet_{dataset_name}.csv", "w") as f:
        f.write(Data)
