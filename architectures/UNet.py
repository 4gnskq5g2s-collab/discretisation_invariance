import os
import sys
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

class UNet(eqx.Module):
    downsampling: list
    upsampling: list
    pre: list
    post: list
    encoder_decoder: list

    def __init__(self, key, depth=4, N_start=32, kernel_size=3, N_in=2, N_out=1, s1=1.0, s2=1.0):
        padding = kernel_size // 2
        keys = random.split(key, 3)
        self.encoder_decoder = [
            normalize_conv(eqx.nn.Conv(1, in_channels=N_in, out_channels=N_start, kernel_size=kernel_size, padding=padding, key=keys[0]), s1=s1, s2=s2),
            normalize_conv(eqx.nn.Conv(1, in_channels=N_start, out_channels=N_out, kernel_size=kernel_size, padding=padding, key=keys[1]), s1=s1, s2=s2)
        ]
        keys = random.split(key, depth)
        N = N_start
        downsampling = []
        upsampling = []
        pre = []
        post = []
        for key in keys[:-1]:
            keys_ = random.split(key, 4)
            layer_up = normalize_conv(eqx.nn.ConvTranspose(1, in_channels=2*N, out_channels=N, kernel_size=2, stride=2, padding=0, key=keys_[0]), s1=s1, s2=s2)
            layer_down = normalize_conv(eqx.nn.Conv(1, in_channels=N, out_channels=2*N, kernel_size=kernel_size, stride=2, padding=padding, key=keys_[1]), s1=s1, s2=s2)
            layer_pre = normalize_conv(eqx.nn.Conv(1, in_channels=N, out_channels=N, kernel_size=kernel_size, padding=padding, key=keys_[2]), s1=s1, s2=s2)
            layer_post = normalize_conv(eqx.nn.Conv(1, in_channels=2*N, out_channels=N, kernel_size=kernel_size, padding=padding, key=keys_[2]), s1=s1, s2=s2)
            N = 2*N
            upsampling.append(layer_up)
            downsampling.append(layer_down)
            pre.append(layer_pre)
            post.append(layer_post)
        keys_ = random.split(keys[-1])
        layer_pre = normalize_conv(eqx.nn.Conv(1, in_channels=N, out_channels=N, kernel_size=kernel_size, padding=padding, key=keys_[2]), s1=s1, s2=s2)
        layer_post = normalize_conv(eqx.nn.Conv(1, in_channels=N, out_channels=N, kernel_size=kernel_size, padding=padding, key=keys_[2]), s1=s1, s2=s2)
        pre.append(layer_pre)
        post.append(layer_post)

        self.downsampling = downsampling
        self.upsampling = upsampling[::-1]
        self.pre = pre
        self.post = post[::-1]

    def __call__(self, feature, x):
        u = jnp.concatenate([feature, x], 0)
        u = self.encoder_decoder[0](u)
        u = gelu(self.pre[0](u))
        Z = [u,]
        for i in range(len(self.downsampling)):
            u = self.downsampling[i](u)
            u = gelu(self.pre[i+1](u))
            Z.append(u)
        Z = Z[::-1]
        u = gelu(self.post[0](u))
        for i in range(len(self.upsampling)):
            u = self.upsampling[i](u)
            u = jnp.concatenate([u, Z[i+1]], 0)
            u = gelu(self.post[i+1](u))
        u = self.encoder_decoder[1](u)
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

    for adjust in [True, False]:
        depths = [5, 6, 7, 8] if adjust else [5, 5, 5, 5]
        Data = "N_train,N_inference,train_error,test_error"
        for J, depth in zip([3, 2, 1, 0], depths):
            N_drop = 100
            N_epoch = 1000
            N_train = 900
            N_batch = 10
            N_start = 16
            D = features.ndim - 2
            learning_rate = 1e-4
            gamma = 0.5
            key = random.PRNGKey(45)
            keys = random.split(key)

            N_run = N_epoch * N_train // N_batch
            N_drop = N_drop * N_train // N_batch
            
            model = UNet(keys[0], N_start=N_start, depth=depth, s1=1e-2, s2=0)
            model_size = sum(tree_map(lambda x: jnp.size(x) if x.dtype == jnp.float32 else 2*jnp.size(x), tree_flatten(model)[0], is_leaf=eqx.is_array))
            optim = optax.adam(learning_rate=learning_rate)
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
        if dataset_name == 'Burgers':
            save_to = "UNet_adjusted_Burgers.csv" if adjust else "UNet_Burgers.csv"
        else:
            save_to = "UNet_adjusted_diffusion.csv" if adjust else "UNet_diffusion.csv"
        with open(save_to, "w") as f:
            f.write(Data)
