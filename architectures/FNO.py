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

def normalize_conv(A, s1=1.0, s2=1.0):
    A = eqx.tree_at(lambda x: x.weight, A, A.weight * s1)
    try:
        A = eqx.tree_at(lambda x: x.bias, A, A.bias * s2)
    except:
        pass
    return A

class FFNO(eqx.Module):
    encoder: eqx.Module
    decoder: eqx.Module
    convs1: list
    convs2: list
    A: jnp.array

    def __init__(self, N_layers, N_features, N_modes, D, key, s1=1.0, s2=1.0, s3=1.0):
        n_in, n_processor, n_out = N_features

        keys = random.split(key, 3 + 2*N_layers)
        self.encoder = normalize_conv(eqx.nn.Conv(D, n_in, n_processor, 1, key=keys[-1]), s1=s1, s2=s2)
        self.decoder = normalize_conv(eqx.nn.Conv(D, n_processor, n_out, 1, key=keys[-2]), s1=s1, s2=s2)
        self.convs1 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key), s1=s1, s2=s2) for key in keys[:N_layers]]
        self.convs2 = [normalize_conv(eqx.nn.Conv(D, n_processor, n_processor, 1, key=key), s1=s1, s2=s2) for key in keys[N_layers:2*N_layers]]
        self.A = random.normal(keys[-3], [N_layers, n_processor, n_processor, N_modes, D], dtype=jnp.complex64) * s3

    def __call__(self, u, x):
        u = jnp.concatenate([x, u], 0)
        u = self.encoder(u)
        for conv1, conv2, A in zip(self.convs1, self.convs2, self.A):
            u += gelu(conv2(gelu(conv1(self.spectral_conv(u, A)))))
        u = self.decoder(u)
        return u

    def spectral_conv(self, v, A):
        u = 0
        N = v.shape
        for i in range(A.shape[-1]):
            u_ = dynamic_slice_in_dim(jnp.fft.rfft(v, axis=i+1), 0, A.shape[-2], axis=i+1)
            u_ = dot_general(A[:, :, :, i], u_, (((1,), (0,)), ((2, ), (i+1, ))))
            u_ = jnp.moveaxis(u_, 0, i+1)
            u += jnp.fft.irfft(u_, axis=i+1, n=N[i+1])
        return u

def l2_loss(model, input, target, x):
    X = model(input, x)
    error = jnp.mean(jnp.sum(((X - target).reshape(target.shape[0], -1,))**2, axis=1))
    return error

def batch_l2_loss(model, input, target, x):
    res = vmap(l2_loss, in_axes=(None, 0, 0, None))(model, input, target, x)
    return jnp.mean(res)

l2_compute_loss_and_grads = eqx.filter_value_and_grad(batch_l2_loss)

def l2_make_step_scan(carry, n, optim):
    model, features, targets, x, opt_state = carry
    loss, grads = l2_compute_loss_and_grads(model, features[n], targets[n], x)
    grads = tree_map(lambda x: x.conj(), grads)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, features, targets, x, opt_state], loss

def make_prediction_scan(carry, i):
    model, features, coords = carry
    prediction = model(features[i], coords)
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
        N_processor = 64
        N_layers = 4
        N_modes = 16
        N_epoch = 400
        N_drop = 100
        N_train = 900
        N_batch = 10
        D = features.ndim - 2
        learning_rate = 1e-4
        gamma = 0.5
        key = random.PRNGKey(45)
        keys = random.split(key)

        N_run = N_epoch * N_train // N_batch
        N_drop = N_drop * N_train // N_batch

        N_features = [coordinates.shape[0] + features.shape[1], N_processor, targets.shape[1]]
        model = FFNO(N_layers, N_features, N_modes, D, keys[0], s1=1e-1, s2=0, s3=1e-1)
        model_size = sum(tree_map(lambda x: jnp.size(x) if x.dtype == jnp.float32 else 2*jnp.size(x), tree_flatten(model)[0], is_leaf=eqx.is_array))
        #learning_rate_ = optax.exponential_decay(learning_rate, N_drop, gamma)
        optim = optax.lion(learning_rate=learning_rate)
        opt_state = optim.init(eqx.filter(model, eqx.is_array))
        n = random.choice(keys[1], N_train, shape = (N_run, N_batch))
        carry = [model, features[:, :, ::2**J], targets[:, :, ::2**J], coordinates[:, ::2**J], opt_state]

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
            predictions = scan(make_prediction_scan, [model, features[:, :, ::2**j], coordinates[:, ::2**j]], ind)[1]
            errors = jnp.linalg.norm(predictions - targets[:, :, ::2**j], axis=2) / jnp.linalg.norm(targets[:, :, ::2**j], axis=2)
            train_error = jnp.mean(errors[:N_train])
            test_error = jnp.mean(errors[N_train:])
            print(f"N_x_train = {N_x_train}, N_x_inference = {N_x}, train error {train_error}, test error {test_error}")
            Data += f"\n{N_x_train},{N_x},{train_error},{test_error}"

    with open(f"FNO_{dataset_name}.csv", "w") as f:
        f.write(Data)
