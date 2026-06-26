import argparse
import itertools
import os

import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd

import run as mlp_run


TARGET_DIM = 2


def build_train_args(base_args, hidden_size, hidden_layers, drift_lr, diffusion_lr, minibatch_size):
    return argparse.Namespace(
        env_name="Rossler",
        extra="",
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        lr=None,
        drift_lr=drift_lr,
        diffusion_lr=diffusion_lr,
        epochs=base_args.epochs,
        minibatch_size=minibatch_size,
        patience=base_args.patience,
        min_sigma=base_args.min_sigma,
        eps=base_args.eps,
        print_every=base_args.print_every,
        num_seeds=base_args.num_seeds,
        target_dim=TARGET_DIM,
        output_dir=base_args.output_dir,
    )


def evaluate_config(base_args, hidden_size, hidden_layers, drift_lr, diffusion_lr, minibatch_size):
    env, dt, horizon, _, _ = mlp_run.infer_experiment_setup("Rossler", "")

    test_ts, test_ys = mlp_run.generate_data(jr.PRNGKey(101), env, 0.01, horizon, 16)
    test_grid = test_ys.reshape(-1, test_ys.shape[-1])
    test_drift = jax.vmap(lambda x: env.drift(0.0, x, jnp.array([0.0])))(test_grid)
    test_diffusion = mlp_run.diffusion_diag_at_grid(env, test_grid)

    config_args = build_train_args(
        base_args,
        hidden_size=hidden_size,
        hidden_layers=hidden_layers,
        drift_lr=drift_lr,
        diffusion_lr=diffusion_lr,
        minibatch_size=minibatch_size,
    )

    rows = []
    for seed in range(base_args.num_seeds):
        key = jr.PRNGKey(seed)
        data_key, val_key, train_key = jr.split(key, 3)
        train_ts, train_ys = mlp_run.generate_data(data_key, env, dt, horizon, 8)
        val_ts, val_ys = mlp_run.generate_data(val_key, env, dt, horizon, 8)

        dim_key = jr.fold_in(train_key, TARGET_DIM)
        best_drift_params, best_diffusion_params, data_stats = mlp_run.train_one_seed(
            env,
            train_ys,
            train_ts,
            val_ys,
            val_ts,
            config_args,
            dim_key,
            TARGET_DIM,
        )

        x_mean, x_std, drift_scale = data_stats
        test_grid_norm = (test_grid - x_mean) / x_std
        drift_pred = jax.vmap(lambda x: mlp_run.mlp_forward(best_drift_params, x))(test_grid_norm)[:, 0] * float(drift_scale)
        diffusion_pred = jax.vmap(
            lambda x: jnp.exp(mlp_run.mlp_forward(best_diffusion_params, x))
        )(test_grid_norm)[:, 0]

        test_drift_mse = jnp.mean((drift_pred - test_drift[:, TARGET_DIM]) ** 2)
        test_diffusion_mse = jnp.mean(
            (diffusion_pred - jnp.abs(test_diffusion[:, TARGET_DIM])) ** 2
        )
        print(test_drift_mse, test_diffusion_mse)

        rows.append(
            {
                "seed": seed,
                "target_dim": TARGET_DIM,
                "hidden_size": hidden_size,
                "hidden_layers": hidden_layers,
                "drift_lr": drift_lr,
                "diffusion_lr": diffusion_lr,
                "minibatch_size": minibatch_size,
                "epochs": base_args.epochs,
                "patience": base_args.patience,
                "test_drift_mse": float(test_drift_mse),
                "test_diffusion_mse": float(test_diffusion_mse),
                "score": float(test_drift_mse + test_diffusion_mse),
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter sweep for the MLP baseline on Rossler target dim 2"
    )
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--hidden_layers", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--drift_lrs", type=float, nargs="+", default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--diffusion_lrs", type=float, nargs="+", default=[1e-6, 1e-5, 1e-4])
    parser.add_argument("--minibatch_sizes", type=int, nargs="+", default=[512])
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=1000)
    parser.add_argument("--min_sigma", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--num_seeds", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="data/MLP_SDE_sweeps")
    parser.add_argument("--output_name", type=str, default="Rossler_x2_sweep")
    args = parser.parse_args()

    all_rows = []
    configs = list(
        itertools.product(
            args.hidden_sizes,
            args.hidden_layers,
            args.drift_lrs,
            args.diffusion_lrs,
            args.minibatch_sizes,
        )
    )

    os.makedirs(args.output_dir, exist_ok=True)
    detail_path = os.path.join(args.output_dir, f"{args.output_name}_detail.csv")

    print(f"Running {len(configs)} configurations for Rossler target dim {TARGET_DIM}")
    for config_index, (hidden_size, hidden_layers, drift_lr, diffusion_lr, minibatch_size) in enumerate(configs, start=1):
        print(
            f"\n[{config_index}/{len(configs)}] hidden_size={hidden_size}, "
            f"hidden_layers={hidden_layers}, drift_lr={drift_lr}, "
            f"diffusion_lr={diffusion_lr}, minibatch_size={minibatch_size}"
        )

        config_rows = evaluate_config(
            args,
            hidden_size=hidden_size,
            hidden_layers=hidden_layers,
            drift_lr=drift_lr,
            diffusion_lr=diffusion_lr,
            minibatch_size=minibatch_size,
        )
        all_rows.extend(config_rows)

        mean_score = sum(row["score"] for row in config_rows) / len(config_rows)
        mean_drift = sum(row["test_drift_mse"] for row in config_rows) / len(config_rows)
        mean_diffusion = sum(row["test_diffusion_mse"] for row in config_rows) / len(config_rows)
        print(
            f"Mean drift MSE={mean_drift:.6f}, mean diffusion MSE={mean_diffusion:.6f}, "
            f"mean score={mean_score:.6f}"
        )

        # Save all results accumulated so far after every config
        pd.DataFrame(all_rows).to_csv(detail_path, index=False)
        print(f"Results saved ({len(all_rows)} rows) → {detail_path}")

    df = pd.DataFrame(all_rows)
    summary_df = (
        df.groupby(["hidden_size", "hidden_layers", "drift_lr", "diffusion_lr", "minibatch_size"], as_index=False)
        .agg(
            mean_test_drift_mse=("test_drift_mse", "mean"),
            std_test_drift_mse=("test_drift_mse", "std"),
            mean_test_diffusion_mse=("test_diffusion_mse", "mean"),
            std_test_diffusion_mse=("test_diffusion_mse", "std"),
            mean_score=("score", "mean"),
            std_score=("score", "std"),
        )
        .sort_values("mean_score")
    )

    summary_path = os.path.join(args.output_dir, f"{args.output_name}_summary.csv")
    df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    # print(f"\nDetailed results saved to: {detail_path}")
    # print(f"Summary results saved to: {summary_path}")
    if not summary_df.empty:
        best = summary_df.iloc[0]
        print("Best configuration:")
        print(best.to_dict())


if __name__ == "__main__":
    main()