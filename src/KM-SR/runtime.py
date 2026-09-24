import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import jax.random as jr

import pandas as pd
import os
import time
from utils.SDE_envs import Lorenz96
from utils.data_generator import generate_data
from KM_SR import make_library, kramers_moyal_coefficients_multidim, make_equation, sequential_lasso_selection

def eval_drift_param(target_dim, degree, train_ys, train_ts, num_bins=5, alpha=0.01, threshold=0.01, min_bin_size=5):    
    """Train on train_ys/train_ts and evaluate on eval_ys"""
    # Compute KM coefficients on training data
    drift_coefficients, _, grid_points = kramers_moyal_coefficients_multidim(
        train_ys, train_ts, num_bins=num_bins, target_dim=target_dim, min_bin_size=min_bin_size)
    
    # Prepare features for sparse regression
    X_drift = grid_points[~jnp.isnan(drift_coefficients)]
    y_drift = drift_coefficients[~jnp.isnan(drift_coefficients)]

    drift_library, drift_names = make_library(X_drift, degree)

    # Apply Lasso regression for drift only (trained on training data)
    lasso_drift, lasso_drift_features, drift_names = sequential_lasso_selection(drift_library, y_drift, drift_names, alpha=alpha, threshold=threshold)

    return 0, make_equation(lasso_drift.coef_, drift_names), lasso_drift, lasso_drift_features

def eval_diffusion_param(target_dim, degree, train_ys, train_ts, num_bins=5, alpha=0.01, threshold=0.01, min_bin_size=5):    
    """Train on train_ys/train_ts and evaluate on eval_ys"""
    # Compute KM coefficients on training data
    _, diffusion_coefficients, grid_points = kramers_moyal_coefficients_multidim(
        train_ys, train_ts, num_bins=num_bins, target_dim=target_dim, min_bin_size=min_bin_size)

    # Create polynomial features for diffusion (multiplicative noise form)
    X_diffusion = grid_points[~jnp.isnan(diffusion_coefficients)]
    y_diffusion = diffusion_coefficients[~jnp.isnan(diffusion_coefficients)]

    diffusion_library, diffusion_names = make_library(X_diffusion, degree, absolute=True)
    # Apply Lasso regression for diffusion only (trained on training data)
    lasso_diffusion, lasso_diffusion_features, diffusion_names = sequential_lasso_selection(diffusion_library, y_diffusion, diffusion_names, alpha=alpha, threshold=threshold)

    return 0, make_equation(lasso_diffusion.coef_, diffusion_names), lasso_diffusion, lasso_diffusion_features

def sparse_regression(ts, ys, target_dim):
    
    mses = []
    equations = []
    params = []
    models = []  # Store models for later testing
    features = []  # Store features for later testing
    degree = 2
    bins = [4]
    
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

    return best_drift, best_diffusion

if __name__ == '__main__':
    batch_size = 8

    N_var = int(sys.argv[1])
    dt = 0.02
    ts_type = "normal"

    noise_level = 0.2
    env = Lorenz96(N_var, noise_level, 4)
    T = 25

    times = []

    for seed in range(11):
        key = jr.PRNGKey(seed)
        data_key, val_data_key, gp_key = jr.split(key, 3)
        ts, ys = generate_data(data_key, env, dt, T, batch_size)

        # Initialize result dictionary for this seed
        seed_result = {
            'seed': seed,
        }
        print(seed)

        start = time.time()

        N = 1
        
        for target_dim in range(N):
            _ = sparse_regression(ts, ys, target_dim)
        
        # Add this seed's results to the main results list
        end = time.time()

        times.append(end-start)

    # Save to CSV
    jnp.save(os.path.join("GP-SDE/data/runtimes", f"KM_SR_{N_var}") + "_time.npy", jnp.array(times))