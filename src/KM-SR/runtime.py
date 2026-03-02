import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(1, '../')

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

    return make_equation(lasso_drift.coef_, drift_names)

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

    return make_equation(lasso_diffusion.coef_, diffusion_names)

def sparse_regression(ts, ys, target_dim):
    
    bins = [16]
    degree = 2
    
    for num_bins in bins:
        for alpha in [0.001, 0.01, 0.1]:
            for threshold in [0.05, 0.1, 0.2]:
                for min_bin_size in [1, 10, 20]:
                        # Train on training data, evaluate on validation data
                        try:
                            mse = eval_drift_param(
                                target_dim, degree=degree, train_ys=ys, train_ts=ts,
                                num_bins=num_bins, alpha=alpha, threshold=threshold, min_bin_size=min_bin_size
                            )
                        except:
                             pass
    
    for num_bins in bins:
        for alpha in [0.001, 0.01, 0.1]:
            for threshold in [0.05, 0.1, 0.2]:
                for min_bin_size in [1, 10, 20]:
                        try:
                            mse = eval_diffusion_param(
                                target_dim, degree=degree, train_ys=ys, train_ts=ts,
                                num_bins=num_bins, alpha=alpha, threshold=threshold, min_bin_size=min_bin_size
                            )
                        except:
                             pass
    return mse

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
        ts, ys = generate_data(data_key, env, dt, T, batch_size, ts_type = ts_type)

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
    jnp.save(os.path.join("/home/sdevries/results", f"SDEs/runtimes/KM_SR_{N_var}_16") + "_time.npy", jnp.array(times))