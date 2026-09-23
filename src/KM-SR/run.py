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

def sparse_regression(ts, ys, test_grid, test_drift, test_diffusion, target_dim):
    
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
                    try:
                        # Train on training data, evaluate on validation data
                        mse, equation, model, feats = eval_drift_param(
                            target_dim, degree=degree, train_ys=ys, train_ts=ts, 
                            num_bins=num_bins, alpha=alpha, threshold=threshold, min_bin_size=min_bin_size
                        )
                        mses.append(mse)
                        equations.append(equation)
                        params.append((num_bins, alpha, threshold, min_bin_size))
                        models.append(model)
                        features.append(feats)
                    except:
                        pass
    
    # Keep drift candidates in lists: models, features, equations, params
    drift_models = models
    drift_features = features
    drift_equations = equations
    drift_params = params

    # Reset containers for diffusion search
    mses = []
    equations = []
    params = []
    models = []  # Store models for later testing
    features = []  # Store features for later testing

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
                            num_bins=num_bins, alpha=alpha, threshold=threshold, min_bin_size=min_bin_size
                        )
                        mses.append(mse)
                        equations.append(equation)
                        params.append((num_bins, alpha, threshold, min_bin_size))
                        models.append(model)
                        features.append(feats)
                    except:
                        pass

    diffusion_models = models
    diffusion_features = features
    diffusion_equations = equations
    diffusion_params = params

    # Use training increments to score all drift-diffusion model pairs via MDL (complexity + NLL)
    # Prepare training transitions from ys
    x_t = ys[:, :-1, :]
    x_tp1 = ys[:, 1:, :]
    dx = x_tp1 - x_t
    x_t_grid = x_t.reshape(x_t.shape[0] * x_t.shape[1], x_t.shape[2])
    dx_grid = dx.reshape(dx.shape[0] * dx.shape[1], dx.shape[2])

    # Precompute feature libraries for the training states
    drift_lib, _ = make_library(x_t_grid, degree)
    diffusion_lib, _ = make_library(x_t_grid, degree, absolute=True)

    M_d = drift_lib.shape[1]
    M_s = diffusion_lib.shape[1]

    eps = 1e-9

    best_score = jnp.inf
    best_pair = (0, 0)

    # Complexity metric: number of selected features
    drift_complexities = [int(jnp.sum(f)) for f in drift_features]
    diffusion_complexities = [int(jnp.sum(f)) for f in diffusion_features]

    for i, d_model in enumerate(drift_models):
        # predict drift for training states
        feat_idx = jnp.where(drift_features[i])[0]
        if feat_idx.size == 0:
            f_pred = jnp.zeros((x_t_grid.shape[0],))
        else:
            f_pred = jnp.array(d_model.predict(drift_lib[:, feat_idx]))

        for j, s_model in enumerate(diffusion_models):
            s_feat_idx = jnp.where(diffusion_features[j])[0]
            if s_feat_idx.size == 0:
                sigma_pred = jnp.ones((x_t_grid.shape[0],)) * 1e-3
            else:
                sigma_pred = jnp.abs(jnp.array(s_model.predict(diffusion_lib[:, s_feat_idx])))

            # compute NLL on training increments for target_dim only
            mu = x_t_grid[:, target_dim] + f_pred * (ts[0,1] - ts[0,0])
            var = (sigma_pred ** 2) * (ts[0,1] - ts[0,0])
            var = jnp.clip(var, min=1e-5)
            residual = x_tp1.reshape(-1, x_tp1.shape[2])[:, target_dim] - mu
            nll_per = 0.5 * (jnp.log(2 * jnp.pi * var) + (residual ** 2) / var)
            total_nll = jnp.sum(nll_per)

            mdl = drift_complexities[i] * jnp.log(M_d) + diffusion_complexities[j] * jnp.log(M_s) + total_nll

            if mdl < best_score:
                best_score = mdl
                best_pair = (i, j)

    best_d_idx, best_s_idx = best_pair
    best_drift = drift_equations[best_d_idx]
    best_diffusion = diffusion_equations[best_s_idx]

    # Evaluate selected pair on test data
    lib_test_d, _ = make_library(test_grid, degree)
    lib_test_s, _ = make_library(test_grid, degree, absolute=True)
    d_feat_idx = jnp.where(drift_features[best_d_idx])[0]
    s_feat_idx = jnp.where(diffusion_features[best_s_idx])[0]

    if d_feat_idx.size == 0:
        preds_d = jnp.zeros((lib_test_d.shape[0],))
    else:
        preds_d = jnp.array(drift_models[best_d_idx].predict(lib_test_d[:, d_feat_idx]))

    if s_feat_idx.size == 0:
        preds_s = jnp.ones((lib_test_s.shape[0],)) * 1e-3
    else:
        preds_s = jnp.abs(jnp.array(diffusion_models[best_s_idx].predict(lib_test_s[:, s_feat_idx])))

    test_drift_mse = jnp.mean((test_drift - preds_d) ** 2)
    test_diffusion_mse = jnp.mean((test_diffusion - preds_s) ** 2)

    print(f"Selected by MDL (train increments): drift={best_drift}, diffusion={best_diffusion}, score={float(best_score)}")

    return best_drift, best_diffusion, float(test_drift_mse), float(test_diffusion_mse)

if __name__ == '__main__':
    batch_size = 8

    env_name = sys.argv[1]
    obs_noise = float(sys.argv[3]) if len(sys.argv)>3 else 0.0

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
        dt = sys.argv[2] if len(sys.argv) > 2 else 0.02
        save_path = f"LV_{dt}"
        dt = jnp.float32(dt)

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

    elif env_name=="vanderPol":
        noise_level = 0.2
        N_var = 2
        env = VanDerPolOscillator(noise_level)
        T = 50
        dt = 0.02
        save_path = f"vdPol"

    # Initialize list to collect results
    results = []
    times = []

    test_ts, test_ys = generate_data(jr.PRNGKey(101), env, 0.01, T, 16)
    test_grid = test_ys.reshape(test_ys.shape[0] * test_ys.shape[1], test_ys.shape[2])
    test_drift = jax.vmap(lambda x: env.drift(0, x, jnp.array([0])))(test_grid)
    test_diffusion = jax.vmap(lambda x: env.diffusion(0, x, jnp.array([0])))(test_grid)

    for seed in range(10):
        key = jr.PRNGKey(seed)
        data_key, gp_key = jr.split(key)
        ts, ys = generate_data(data_key, env, dt, T, batch_size, obs_sigma=obs_noise)
        

        # Initialize result dictionary for this seed
        seed_result = {
            'seed': seed,
        }
        print("seed:", seed)

        N = 1 if env_name == "Lorenz96" else env.n_var
        
        for target_dim in range(N):
            _test_drift = test_drift[:, target_dim]

            _test_diffusion = test_diffusion[:, target_dim, target_dim]

            drift_eq, diffusion_eq, test_drift_mse, test_diffusion_mse = sparse_regression(ts, ys, test_grid, _test_drift, _test_diffusion, target_dim)

            seed_result[f'x{target_dim}_equation'] = f"[{drift_eq}, {diffusion_eq}]"
            seed_result[f'x{target_dim}_test_drift_mse'] = float(test_drift_mse)
            seed_result[f'x{target_dim}_test_diffusion_mse'] = float(test_diffusion_mse)

        results.append(seed_result)

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Create filename based on experiment parameters
    filename = f"GP-SDE/data/KM_SR/{save_path}.csv"

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"\nResults saved to: {filename}")
    print(f"Total experiments completed: {len(results)}")