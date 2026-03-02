import warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import jax.random as jr

from itertools import combinations_with_replacement
from sklearn.linear_model import Lasso

def kramers_moyal_coefficients_multidim(ys, ts, num_bins=20, target_dim=0, min_bin_size = 5):
    """
    Kramers-Moyal coefficients for multidimensional state space
    
    Args:
        ys: trajectory data [batch_size, num_points, n_dims]
        ts: time points [batch_size, num_points] 
        num_bins: number of spatial bins per dimension
        taus: list of time lag indices
        target_dim: which dimension to analyze as output
    """

    tau = 1
    
    if ys.ndim != 3:
        raise ValueError(f"Expected 3D input [batch_size, num_points, n_dims], got {ys.ndim}D")
    
    batch_size, num_points, n_dims = ys.shape
    index_array = jnp.arange(num_points)
    
    # Target variable (output)
    y_target = ys[:, :, target_dim]  # [batch_size, num_points]
    
    # Create bins for each dimension
    bins = jnp.zeros((n_dims, num_bins))
    for dim in range(n_dims):
        dim_data = ys[:, :, dim].flatten()
        valid_data = dim_data[jnp.isfinite(dim_data)]
        percentiles = jnp.percentile(valid_data, jnp.array([10, 90]))
        bins = bins.at[dim].set(jnp.linspace(percentiles[0], percentiles[1], num_bins))
    
    # Create multidimensional grid
    if n_dims == 1:
        grid_points = bins[0][:, None]  # Shape: [num_bins, 1]
    elif n_dims == 2:
        X0, X1 = jnp.meshgrid(bins[0], bins[1], indexing='ij')
        grid_points = jnp.stack([X0.flatten(), X1.flatten()], axis=1)
    elif n_dims == 3:
        X0, X1, X2 = jnp.meshgrid(bins[0], bins[1], bins[2], indexing='ij')
        grid_points = jnp.stack([X0.flatten(), X1.flatten(), X2.flatten()], axis=1)
    elif n_dims > 3:
        XS = jnp.meshgrid(*bins, indexing='ij')
        grid_points = jnp.stack([_X.flatten() for _X in XS], axis=1)
    else:
        # For higher dimensions, use a subset or different approach
        raise NotImplementedError(f"Plotting for {n_dims}D not implemented")
    
    def process_batch(batch_ys, batch_ts, batch_y_target):
       
        def process_grid(grid_point):
            # Find points near this grid location
            # Create bin mask for each dimension
            bin_width_per_dim = jnp.array([jnp.mean(jnp.diff(bins[dim])) * 1.5 for dim in range(n_dims)])
            bin_mask = jnp.ones(batch_ys.shape[0], dtype=bool)

            for dim in range(n_dims):
                dim_mask = jnp.abs(batch_ys[:, dim] - grid_point[dim]) < bin_width_per_dim[dim] / 2
                bin_mask = bin_mask & dim_mask
                                        
            # Calculate increments for target dimension
            deltas = (jnp.roll(batch_y_target, -tau) - batch_y_target)
            dt_array = (jnp.roll(batch_ts, -tau) - batch_ts)
            
            normalized_deltas = deltas / dt_array
            normalized_deltas_sq = (deltas**2) / dt_array
            
            drift = jnp.where(bin_mask & (index_array < (num_points - tau)), normalized_deltas, jnp.nan)
            diffusion = jnp.where(bin_mask & (index_array < (num_points - tau)), normalized_deltas_sq, jnp.nan)
            
            drift = jax.lax.select(jnp.sum(bin_mask) > min_bin_size, jnp.nanmean(drift), jnp.nan)
            diffusion = jax.lax.select(jnp.sum(bin_mask) > min_bin_size, jnp.nanmean(diffusion), jnp.nan)

            return jnp.array([drift, diffusion])

        grid_coefficients = jax.vmap(process_grid)(grid_points)
        
        return grid_coefficients
    
    # Process all batches
    all_coefficients = jax.vmap(process_batch)(ys, ts, y_target)
    
    # Average across batches
    drift_coeffs = jnp.nanmean(all_coefficients[:, :, 0], axis=0)
    diffusion_coeffs = jnp.nanmean(all_coefficients[:, :, 1], axis=0)
    
    return drift_coeffs, jnp.sqrt(jnp.abs(diffusion_coeffs)), grid_points

def sequential_lasso_selection(X, y, names, alpha=0.01, threshold = 0.05):
    """
    Sequential forward selection with LASSO
    """
    # Base LASSO estimator
    change = True
    selected_features = jnp.ones(X.shape[-1], dtype=bool)
    _X = X
    temp_length = X.shape[-1]

    while change:
        lasso_base = Lasso(alpha=alpha, fit_intercept=False)
        lasso_base.fit(_X, y)

        # Get features with coefficients above threshold
        current_selected = jnp.abs(lasso_base.coef_) > threshold
        
        # Update global selection mask
        selected_indices = jnp.where(selected_features)[0]
        new_selected_features = jnp.zeros(X.shape[-1], dtype=bool)
        new_selected_features = new_selected_features.at[selected_indices[current_selected]].set(True)
        
        # Check for convergence
        if jnp.sum(new_selected_features) == temp_length:
            change = False
        else:
            selected_features = new_selected_features
            _X = X[:, selected_features]
            temp_length = jnp.sum(selected_features)
    
        # Handle the case where no features are selected
        if jnp.sum(selected_features) == 0:
            # Select only the constant term (first feature)
            selected_features = selected_features.at[0].set(True)
            lasso_base = Lasso(alpha=alpha, fit_intercept=False)
            lasso_base.fit(X[:, :1], y)  # Only use the constant term
            return lasso_base, selected_features, [names[0]]  # Return only the constant term name
    
    return lasso_base, selected_features, [names[i] for i in range(len(names)) if bool(selected_features[i])]

def make_library(X, degree, absolute = False):
    if len(X.shape)==1:
        X = X[:,None]

    if X.shape[-1]==1:
        terms = [X.flatten()**d for d in range(degree + 1)]
        names = ["1"] + [f"x**{i+1}" for i in range(degree)]
        if absolute:
            terms.append(jnp.abs(X.flatten()))
            names.append("|x|")
        return jnp.column_stack(terms), names
    # Generate all polynomial terms up to degree
    terms = []
    features = ["1"]
    variables = ["x" + str(i) for i in range(X.shape[-1])]
    n_vars = X.shape[-1]

    # Add constant term
    terms.append(jnp.ones((X.shape[0], 1)))

    # Generate all combinations of powers for each variable  
    for d in range(degree):
        for comb in combinations_with_replacement(jnp.arange(n_vars), d+1):
            term = jnp.ones_like(X[:,0])
            name = ""
            for x in comb:
                term *= X[:,x]
                name = name + variables[x]
            terms.append(term)
            features.append(name)

    if absolute:
        for i in range(n_vars):
            terms.append(jnp.abs(X[:, i]))
            features.append(f"|{variables[i]}|")

    return jnp.column_stack(terms), features

def make_equation(coef, names):
    if names[0] == "1":
        equation = f"{coef[0]:.3f}"
    else:
        equation = f"{coef[0]:.3f}{names[0]}"

    for c, n in zip(coef[1:], names[1:]):
        if c < 0:
            equation += f" {c:.3f}{n}"
        else:
            equation += f" + {c:.3f}{n}"

    return equation

def eval_drift_param(target_dim, degree, train_ys, train_ts, val_grid, val_targets, num_bins=5, alpha=0.01, threshold=0.01, min_bin_size=5):    
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
    
    # Evaluate on provided evaluation dataset
    library, _ = make_library(val_grid, degree)
    library = library[:, lasso_drift_features]
    preds = lasso_drift.predict(library)
    val_mse = jnp.mean((val_targets - preds) ** 2)

    return val_mse, make_equation(lasso_drift.coef_, drift_names), lasso_drift, lasso_drift_features

def eval_diffusion_param(target_dim, degree, train_ys, train_ts, val_grid, val_targets, num_bins=5, alpha=0.01, threshold=0.01, min_bin_size=5):    
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
    
    # Evaluate on provided evaluation dataset
    library, _ = make_library(val_grid, degree, absolute=True)
    library = library[:, lasso_diffusion_features]
    preds = jnp.abs(lasso_diffusion.predict(library))
    val_mse = jnp.mean((val_targets - preds) ** 2)

    return val_mse, make_equation(lasso_diffusion.coef_, diffusion_names), lasso_diffusion, lasso_diffusion_features