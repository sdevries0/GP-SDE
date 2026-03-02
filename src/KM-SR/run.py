import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import jax.random as jr

import sys
import pandas as pd
import os
import time
from utils.SDE_envs import DoubleWell, LotkaVolterra, VanDerPolOscillator, RosslerAttractor, Lorenz96
from utils.data_generator import generate_data
from KM_SR import make_library, eval_drift_param, eval_diffusion_param

def sparse_regression(ts, ys, val_grid, val_drift, val_diffusion, test_grid, test_drift, test_diffusion, target_dim):
    
    print(f"=== DRIFT{target_dim} HYPERPARAMETER SEARCH ===")
    mses = []
    equations = []
    params = []
    models = []  # Store models for later testing
    features = []  # Store features for later testing
    if ys.shape[-1]==20:
        bins = [2]
        degree = 2
    elif ys.shape[-1]==10:
        bins = [4]
        degree = 2
    elif ys.shape[-1]==5:
        bins = [5, 10, 15]
        degree = 2
    elif ys.shape[-1]==3:
        bins = [5, 10, 25]
        degree = 2
    elif ys.shape[-1]==2:
        bins = [5, 10, 25, 50, 100]
        degree = 3
    elif ys.shape[-1]==1:
        bins = [5, 10, 25, 50, 100]
        degree = 3
    
    for num_bins in bins:
        for alpha in [0.001, 0.01, 0.1]:
            for threshold in [0.05, 0.1, 0.2]:
                for min_bin_size in [1, 10, 20]:
                    # try:
                        # Train on training data, evaluate on validation data
                        mse, equation, model, feats = eval_drift_param(
                            target_dim, degree=degree, train_ys=ys, train_ts=ts, 
                            val_grid=val_grid, val_targets=val_drift,
                            num_bins=num_bins, alpha=alpha, threshold=threshold, min_bin_size=min_bin_size
                        )
                        mses.append(mse)
                        equations.append(equation)
                        params.append((num_bins, alpha, threshold, min_bin_size))
                        models.append(model)
                        features.append(feats)
                    # except:
                    #     pass
    
    best_idx = jnp.argmin(jnp.array(mses))

    best_drift = equations[best_idx]
    
    # Evaluate best model on test data
    library, _ = make_library(test_grid, degree=degree)
    library = library[:, features[best_idx]]
    preds = models[best_idx].predict(library)
    test_drift_mse = jnp.mean((test_drift - preds) ** 2)

    print(f"Best drift (validation): equation = {equations[best_idx]}, validation MSE = {mses[best_idx]}, test MSE = {test_drift_mse}, hyperparameters = {params[best_idx]}")
    print()

    print(f"=== DIFFUSION{target_dim} HYPERPARAMETER SEARCH ===")
    mses = []
    equations = []
    params = []
    models = []  # Store models for later testing
    features = []  # Store features for later testing
    
    for num_bins in bins:
        for alpha in [0.001, 0.01, 0.1]:
            for threshold in [0.05, 0.1, 0.2]:
                for min_bin_size in [1, 10, 20]:
                    try:
                        # Train on training data, evaluate on validation data
                        mse, equation, model, feats = eval_diffusion_param(
                            target_dim, degree=degree, train_ys=ys, train_ts=ts, 
                            val_grid=val_grid, val_targets=val_diffusion,
                            num_bins=num_bins, alpha=alpha, threshold=threshold, min_bin_size=min_bin_size
                        )
                        mses.append(mse)
                        equations.append(equation)
                        params.append((num_bins, alpha, threshold, min_bin_size))
                        models.append(model)
                        features.append(feats)
                    except:
                        pass

    best_idx = jnp.argmin(jnp.array(mses))

    best_diffusion = equations[best_idx]
    
    # Evaluate best model on test data
    library, _ = make_library(test_grid, degree=degree, absolute=True)
    library = library[:, features[best_idx]]
    preds = models[best_idx].predict(library)
    test_diffusion_mse = jnp.mean((test_diffusion - preds) ** 2)

    print(f"Best diffusion (validation): equation = {equations[best_idx]}, validation MSE = {mses[best_idx]}, test MSE = {test_diffusion_mse}, hyperparameters = {params[best_idx]}")
    print()
    
    return best_drift, best_diffusion, test_drift_mse, test_diffusion_mse

if __name__ == '__main__':
    batch_size = 8

    env_name = sys.argv[1]

    if env_name=="Double well":
        diffusion_name = sys.argv[2] if len(sys.argv) > 2 else "additive"
        N_var = 1
        dt = 0.02
        noise_level = 0.5
        env = DoubleWell(noise_level, diffusion_name)
        T = 50
        save_path = f"DW_{diffusion_name}"

    elif env_name=="Lotka-Volterra":
        N_var = 2
        noise_level = 0.2
        env = LotkaVolterra(noise_level)
        T = 50
        dt = jnp.float32(sys.argv[2]) if len(sys.argv) > 2 else 0.02
        save_path = f"LV_{dt}"

    elif env_name=="Lorenz96":
        noise_level = 0.2
        N_var = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        env = Lorenz96(N_var, noise_level, 4)
        T = 25
        dt = 0.02
        save_path = f"Lorenz_{N_var}"

    elif env_name=="Rossler":
        noise_level = 0.1
        N_var = 3
        env = RosslerAttractor(noise_level)
        T = 50
        dt = 0.02
        save_path = f"Rossler"

    elif env_name==vanderPol:
        noise_level = 0.2
        N_var = 2
        env = VanDerPolOscillator(noise_level)
        T = 50
        dt = 0.02
        save_path = fvanderPol

    # Initialize list to collect results
    results = []
    times = []

    test_ts, test_ys = generate_data(jr.PRNGKey(101), env, 0.01, T, 16)
    test_grid = test_ys.reshape(test_ys.shape[0] * test_ys.shape[1], test_ys.shape[2])
    test_drift = jax.vmap(lambda x: env.drift(0, x, jnp.array([0])))(test_grid)
    test_diffusion = jax.vmap(lambda x: env.diffusion(0, x, jnp.array([0])))(test_grid)

    for seed in range(10):
        key = jr.PRNGKey(seed)
        data_key, val_data_key, gp_key = jr.split(key, 3)
        ts, ys = generate_data(data_key, env, dt, T, batch_size)

        val_ts, val_ys = generate_data(val_data_key, env, dt, T, batch_size)

        val_grid = val_ys.reshape(val_ys.shape[0] * val_ys.shape[1], val_ys.shape[2])

        val_drift = jax.vmap(lambda x: env.drift(0, x, jnp.array([0])))(val_grid)
        val_diffusion = jax.vmap(lambda x: env.diffusion(0, x, jnp.array([0])))(val_grid)

        # Initialize result dictionary for this seed
        seed_result = {
            'seed': seed,
        }
        print("seed:", seed)

        N = 1 if env_name == "Lorenz96" else env.n_var
        
        for target_dim in range(N):
            _val_drift = val_drift[:, target_dim]
            _test_drift = test_drift[:, target_dim]

            _val_diffusion = val_diffusion[:, target_dim, target_dim]
            _test_diffusion = test_diffusion[:, target_dim, target_dim]

            drift_eq, diffusion_eq, test_drift_mse, test_diffusion_mse = sparse_regression(ts, ys, val_grid, _val_drift, _val_diffusion, test_grid, _test_drift, _test_diffusion, target_dim)

            seed_result[f'x{target_dim}_equation'] = f"[{drift_eq}, {diffusion_eq}]"
            seed_result[f'x{target_dim}_test_drift_mse'] = float(test_drift_mse)
            seed_result[f'x{target_dim}_test_diffusion_mse'] = float(test_diffusion_mse)

        results.append(seed_result)

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Create filename based on experiment parameters
    filename = f"data/KM_SR/{save_path}.csv"

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"\nResults saved to: {filename}")
    print(f"Total experiments completed: {len(results)}")