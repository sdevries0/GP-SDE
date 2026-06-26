import argparse
import os
import sys
from typing import Dict

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.SDE_envs import (
    DoubleWell,
    Lorenz96,
    LotkaVolterra,
    RosslerAttractor,
    VanDerPolOscillator,
)
from utils.data_generator import generate_data


def init_mlp_params(
    key: jr.PRNGKey,
    in_dim: int,
    out_dim: int,
    hidden_size: int,
    hidden_layers: int,
    scale: float = 0.1,
):
    """Initialize a simple tanh MLP as a pytree of layer dicts."""
    dims = [in_dim] + [hidden_size] * hidden_layers + [out_dim]
    keys = jr.split(key, len(dims) - 1)
    params = []
    for k, d_in, d_out in zip(keys, dims[:-1], dims[1:]):
        w_key, _ = jr.split(k)
        w = scale * jr.normal(w_key, (d_in, d_out)) / jnp.sqrt(float(d_in))
        b = jnp.zeros((d_out,))
        params.append({"w": w, "b": b})
    return params


def mlp_forward(params, x):
    h = x
    for i, layer in enumerate(params):
        h = h @ layer["w"] + layer["b"]
        if i < len(params) - 1:
            h = jax.nn.silu(h)
    return h


def make_transitions(ts: jnp.ndarray, ys: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Convert trajectory batches into transition tuples (x_t, x_{t+1}, dt)."""
    x_t = ys[:, :-1, :].reshape(-1, ys.shape[-1])
    x_tp1 = ys[:, 1:, :].reshape(-1, ys.shape[-1])
    dt = (ts[:, 1:] - ts[:, :-1]).reshape(-1, 1)

    valid = (
        jnp.isfinite(x_t).all(axis=1)
        & jnp.isfinite(x_tp1).all(axis=1)
        & jnp.isfinite(dt[:, 0])
        & (dt[:, 0] > 0)
    )

    return {
        "x_t": x_t[valid],
        "x_tp1": x_tp1[valid],
        "dt": dt[valid],
    }


def nll_loss(
    drift_params,
    diffusion_params,
    batch: Dict[str, jnp.ndarray],
    eps: float,
    target_dim: int,
    drift_scale: float
):
    x_t, x_tp1, dt = batch["x_t"], batch["x_tp1"], batch["dt"]

    drift = jax.vmap(lambda x: mlp_forward(drift_params, x))(x_t) * drift_scale
    sigma = jax.vmap(lambda x: mlp_forward(diffusion_params, x))(x_t)
    sigma = jnp.exp(sigma)

    mean = x_t[:, target_dim:target_dim + 1] + drift * dt
    var = (sigma ** 2) * dt + eps

    sq_term = ((x_tp1[:, target_dim:target_dim + 1] - mean) ** 2) / var
    log_term = jnp.log(2.0 * jnp.pi * var)
    nll = 0.5 * jnp.sum(log_term + sq_term, axis=-1)
    return jnp.mean(nll)


def train_step(
    drift_params,
    diffusion_params,
    drift_opt_state,
    diffusion_opt_state,
    batch,
    drift_optimizer,
    diffusion_optimizer,
    eps,
    target_dim,
    drift_scale,
):
    def loss_fn(d_params, s_params):
        return nll_loss(d_params, s_params, batch, eps=eps, target_dim=target_dim, drift_scale=drift_scale)

    loss, (drift_grads, diffusion_grads) = jax.value_and_grad(loss_fn, argnums=(0, 1))(drift_params, diffusion_params)

    drift_updates, new_drift_opt_state = drift_optimizer.update(drift_grads, drift_opt_state, drift_params)
    diffusion_updates, new_diff_opt_state = diffusion_optimizer.update(diffusion_grads, diffusion_opt_state, diffusion_params)

    new_drift_params = optax.apply_updates(drift_params, drift_updates)
    new_diff_params = optax.apply_updates(diffusion_params, diffusion_updates)

    return new_drift_params, new_diff_params, new_drift_opt_state, new_diff_opt_state, loss


def diffusion_diag_at_grid(env, grid: jnp.ndarray) -> jnp.ndarray:
    diffusion_true = jax.vmap(lambda x: env.diffusion(0.0, x, jnp.array([0.0])))(grid)

    if diffusion_true.ndim == 3:
        return jnp.diagonal(diffusion_true, axis1=1, axis2=2)
    if diffusion_true.ndim == 2:
        return diffusion_true
    return diffusion_true[:, None]


def get_target_dims(args, num_target_dims: int):
    if args.target_dim is None:
        return list(range(num_target_dims))

    if args.target_dim < 0 or args.target_dim >= num_target_dims:
        raise ValueError(
            f"target_dim must be between 0 and {num_target_dims - 1} for {args.env_name}"
        )

    return [args.target_dim]


def train_one_seed(env, train_ys, train_ts, val_ys, val_ts, args, key, target_dim):
    """Train separate drift + diffusion MLPs for a single target dimension."""
    init_drift_key, init_diff_key, shuffle_key = jr.split(key, 3)

    train_transitions = make_transitions(train_ts, train_ys)
    val_transitions   = make_transitions(val_ts, val_ys)

    # --- add normalization ---
    x_mean = 0.0#jnp.mean(train_transitions["x_t"], axis=0)
    x_std  = 1.0#jnp.std(train_transitions["x_t"],  axis=0) + 1e-6
    dx     = (train_transitions["x_tp1"] - train_transitions["x_t"])
    drift_scale = 1.0#float(jnp.std(dx / train_transitions["dt"], axis=0)[target_dim] + 1e-6)

    def normalize(transitions):
        return {
            "x_t":   (transitions["x_t"]  - x_mean) / x_std,
            "x_tp1": (transitions["x_tp1"] - x_mean) / x_std,
            "dt":    transitions["dt"],
        }

    # train_transitions = normalize(train_transitions)
    # val_transitions   = normalize(val_transitions)

    n_var = int(train_ys.shape[-1])
    drift_params = init_mlp_params(init_drift_key, n_var, 1, args.hidden_size, args.hidden_layers)
    diffusion_params = init_mlp_params(init_diff_key, n_var, 1, args.hidden_size, args.hidden_layers)

    drift_lr = args.drift_lr if args.drift_lr is not None else args.lr
    diffusion_lr = args.diffusion_lr if args.diffusion_lr is not None else args.lr

    n_train = train_transitions["x_t"].shape[0]
    n_steps_per_epoch = max(1, n_train // args.minibatch_size)
    n_usable = n_steps_per_epoch * args.minibatch_size
    total_steps = args.epochs * n_steps_per_epoch
    warmup_steps = 500

    # Pre-batch data into [n_steps_per_epoch, minibatch_size, dim] for scan
    x_t_batched   = train_transitions["x_t"][:n_usable].reshape(n_steps_per_epoch, args.minibatch_size, -1)
    x_tp1_batched = train_transitions["x_tp1"][:n_usable].reshape(n_steps_per_epoch, args.minibatch_size, -1)
    dt_batched    = train_transitions["dt"][:n_usable].reshape(n_steps_per_epoch, args.minibatch_size, 1)

    def make_optimizer(peak_lr):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=peak_lr * 0.005,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=peak_lr * 0.1,
        )
        return optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))

    drift_optimizer = make_optimizer(drift_lr)
    diffusion_optimizer = make_optimizer(diffusion_lr)
    drift_opt_state = drift_optimizer.init(drift_params)
    diffusion_opt_state = diffusion_optimizer.init(diffusion_params)

    def scan_step(carry, batch):
        drift_p, diff_p, drift_os, diff_os = carry
        drift_p, diff_p, drift_os, diff_os, loss = train_step(
            drift_p, diff_p, drift_os, diff_os, batch,
            drift_optimizer, diffusion_optimizer,
            args.eps, target_dim, drift_scale,
        )
        return (drift_p, diff_p, drift_os, diff_os), loss

    # JIT-compile once: runs all minibatch steps of one epoch on the GPU without Python re-entry
    one_epoch_fn = jax.jit(lambda carry, batches: jax.lax.scan(scan_step, carry, batches))

    best_val_nll = jnp.inf
    best_drift_params = drift_params
    best_diffusion_params = diffusion_params
    patience_counter = 0
    carry = (drift_params, diffusion_params, drift_opt_state, diffusion_opt_state)

    for epoch in range(1, args.epochs + 1):
        # Shuffle batch order at the Python level before each epoch
        shuffle_key, perm_key = jr.split(shuffle_key)
        perm = jr.permutation(perm_key, n_steps_per_epoch)
        batches = {
            "x_t":   x_t_batched[perm],
            "x_tp1": x_tp1_batched[perm],
            "dt":    dt_batched[perm],
        }

        carry, _ = one_epoch_fn(carry, batches)
        drift_params, diffusion_params = carry[0], carry[1]

        val_nll = float(nll_loss(
            drift_params, diffusion_params, val_transitions,
            eps=args.eps, target_dim=target_dim, drift_scale=drift_scale,
        ))

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_drift_params = drift_params
            best_diffusion_params = diffusion_params
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    return best_drift_params, best_diffusion_params, (x_mean, x_std, drift_scale)


def infer_experiment_setup(env_name: str, extra: str):
    if env_name == "Double well":
        diffusion_name = extra if extra else "additive"
        env = DoubleWell(0.5, diffusion_name)
        return env, 0.02, 50.0, f"DW_{diffusion_name}", env.n_var

    if env_name == "Lotka-Volterra":
        dt = float(extra) if extra else 0.02
        env = LotkaVolterra(0.2)
        return env, dt, 50.0, f"LV_{dt}", env.n_var

    if env_name == "Lorenz96":
        n_var = int(extra) if extra else 5
        env = Lorenz96(n_var, 0.2, 4)
        return env, 0.02, 25.0, f"Lorenz_{n_var}", 1

    if env_name == "Rossler":
        env = RosslerAttractor(0.1)
        return env, 0.02, 50.0, "Rossler", env.n_var

    if env_name in ["vanderPol", "Van der Pol Oscillator", "vdPol"]:
        env = VanDerPolOscillator(0.2)
        return env, 0.02, 50.0, "vdPol", env.n_var

    raise ValueError("Unsupported environment name.")


def run_experiment(args):
    env, dt, horizon, save_stem, num_target_dims = infer_experiment_setup(args.env_name, args.extra)
    target_dims = get_target_dims(args, num_target_dims)

    test_ts, test_ys = generate_data(jr.PRNGKey(101), env, 0.01, horizon, 16)
    test_grid = test_ys.reshape(-1, test_ys.shape[-1])
    test_drift = jax.vmap(lambda x: env.drift(0.0, x, jnp.array([0.0])))(test_grid)
    test_diffusion = diffusion_diag_at_grid(env, test_grid)

    # print("=" * 80)
    # print("Neural SDE MLE training with GP-SDE-compatible experiment setup")
    # print(f"Environment: {args.env_name}, dimensions: {env.n_var}")
    # print(f"Train dt: {dt}, horizon: {horizon}, seeds: {args.num_seeds}")
    # print(
    #     f"Learning rates: drift={args.drift_lr if args.drift_lr is not None else args.lr}, "
    #     f"diffusion={args.diffusion_lr if args.diffusion_lr is not None else args.lr}"
    # )
    # print("Selection: early stopping on validation NLL")
    # print("One separate drift + diffusion MLP trained per state dimension")
    # print("Final score: test drift/diffusion MSE per dimension")
    # print("=" * 80)

    results = []
    for seed in range(args.num_seeds):
        print(seed)
        key = jr.PRNGKey(seed)
        data_key, val_key, train_key = jr.split(key, 3)
        train_ts, train_ys = generate_data(data_key, env, dt, horizon, 8)
        val_ts, val_ys = generate_data(val_key, env, dt, horizon, 8)

        seed_result = {"seed": seed}
        for target_dim in target_dims:
            # Use a separate key per dimension (mirrors GP-SDE's separate key per dim)
            dim_key = jr.fold_in(train_key, target_dim)
            # print(f"\nSeed {seed}, Target dim {target_dim}")
            best_drift_params, best_diffusion_params, data_stats = train_one_seed(
                env, train_ys, train_ts, val_ys, val_ts, args, dim_key, target_dim
            )

            # Per-dim models output shape (N, 1); squeeze to scalar per grid point
            # drift_pred = jax.vmap(lambda x: mlp_forward(best_drift_params, x))(test_grid)[:, 0]

            x_mean, x_std, drift_scale = data_stats
            test_grid_norm = (test_grid - x_mean) / x_std
            drift_pred = jax.vmap(lambda x: mlp_forward(best_drift_params, x))(test_grid_norm)[:, 0] * float(drift_scale)
            diffusion_pred = jax.vmap(
                lambda x: (mlp_forward(best_diffusion_params, x))
            )(test_grid_norm)[:, 0]

            test_drift_mse = jnp.mean((drift_pred - test_drift[:, target_dim]) ** 2)
            test_diffusion_mse = jnp.mean(
                (diffusion_pred - jnp.abs(test_diffusion[:, target_dim])) ** 2
            )
            seed_result[f"x{target_dim}_equation"] = "MLP"
            seed_result[f"x{target_dim}_test_drift_mse"] = float(test_drift_mse)
            seed_result[f"x{target_dim}_test_diffusion_mse"] = float(test_diffusion_mse)

            print(
                f"Seed {seed}, Target dim {target_dim}: "
                f"drift MSE = {float(test_drift_mse):.6f}, "
                f"diffusion MSE = {float(test_diffusion_mse):.6f}"
            )

        results.append(seed_result)

    df = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    filename = os.path.join(args.output_dir, f"{save_stem}.csv")
    df.to_csv(filename, index=False)
    # print(f"\nResults saved to: {filename}")
    # print(f"Total experiments completed: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Neural SDE MLE baseline with GP-SDE-compatible setup"
    )

    parser.add_argument(
        "env_name",
        type=str,
        choices=["Double well", "Lotka-Volterra", "Lorenz96", "Rossler", "vanderPol"],
    )
    parser.add_argument(
        "extra",
        nargs="?",
        default="",
        help=(
            "Optional experiment argument: diffusion type for Double well, "
            "dt for Lotka-Volterra, n_var for Lorenz96"
        ),
    )

    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--drift_lr", type=float, default=1e-4)
    parser.add_argument("--diffusion_lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--minibatch_size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=1000)

    parser.add_argument("--eps", type=float, default=1e-5)

    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--target_dim", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="data/MLP_SDE")

    run_experiment(parser.parse_args())
