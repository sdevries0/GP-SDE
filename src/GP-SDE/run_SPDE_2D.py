import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import os
from utils.SDE_envs import SPDE_2D

from kozax.genetic_programming import GeneticProgramming
from utils.fitness_functions import FitnessFunctionSPDE_2D
    
def validate(solution, us, target_drift, target_diffusion, tree_evaluator):
    """
    Compute MSE between predicted and true drift/diffusion for each solution in pareto front.
    
    Args:
        pareto_front: List of solutions from the pareto front
        grid: Grid points where to evaluate (reshaped ys data)
        target_drift: True drift values at grid points
        target_diffusion: True diffusion values at grid points
        tree_evaluator: Function to evaluate tree expressions
    
    Returns:
        drift_mses: MSE values for drift predictions
        diffusion_mses: MSE values for diffusion predictions
    """
    # Evaluate solution at all grid points
    d2u_dx2 = (jnp.roll(us, 1, axis=1) - 2*us + jnp.roll(us, -1, axis=1)) * solver.dx2_inv
    d2u_dy2 = (jnp.roll(us, 1, axis=2) - 2*us + jnp.roll(us, -1, axis=2)) * solver.dy2_inv

    du_dx = (jnp.roll(us, -1, axis=1) - jnp.roll(us, 1, axis=1)) / (2 * solver.dx)
    du_dy = (jnp.roll(us, -1, axis=2) - jnp.roll(us, 1, axis=2)) / (2 * solver.dy)

    predictions = vmap_PDE(solution, us, du_dx, du_dy, d2u_dx2, d2u_dy2, tree_evaluator)
    
    # Extract drift and diffusion predictions
    pred_drift = predictions[..., 0]  # First output is drift
    pred_diffusion = jnp.abs(predictions[..., 1])  # Second output is diffusion (sigma)
    
    # Convert sigma to variance for comparison with target_diffusion
    pred_variance = jnp.abs(pred_diffusion)
    
    # Compute MSE for drift
    drift_mse = jnp.mean((pred_drift - target_drift)**2)
    
    # Compute MSE for diffusion
    diffusion_mse = jnp.mean((pred_variance - target_diffusion)**2)
    
    return drift_mse, diffusion_mse
    
def vmap_PDE(candidate, us, du_dx, du_dy, d2u_dx2, d2u_dy2, tree_evaluator):
    vmap_y = jax.vmap(lambda u, du_dx, du_dy, d2udx, d2udy: tree_evaluator(candidate, jnp.array([u, du_dx, du_dy, d2udx + d2udy])), in_axes=[0,0,0,0,0])
    vmap_x = jax.vmap(lambda u, du_dx, du_dy, d2udx, d2udy: vmap_y(u, du_dx, du_dy, d2udx, d2udy), in_axes=[0,0,0,0,0])
    vmap_time = jax.vmap(lambda u, du_dx, du_dy, d2udx, d2udy: vmap_x(u, du_dx, du_dy, d2udx, d2udy), in_axes=[0,0,0,0,0])

    return vmap_time(us, du_dx, du_dy, d2u_dx2, d2u_dy2)

def compute_nll_for_solution_spde(solution, u_t, u_tp1, dt, tree_evaluator):
    """
    Compute total negative log-likelihood for SPDE solution on temporal increments.

    u_t, u_tp1: arrays shape (n_traj, n_time, nx)
    Predictions use local spatial derivatives computed from u_t via solver.
    """
    # compute spatial derivatives on u_t
    # d2u_dx2 and du_dx shapes (n_traj, n_time, nx)
    d2u_dx2 = (jnp.roll(u_t, 1, axis=1) - 2*u_t + jnp.roll(u_t, -1, axis=1)) * solver.dx2_inv
    d2u_dy2 = (jnp.roll(u_t, 1, axis=2) - 2*u_t + jnp.roll(u_t, -1, axis=2)) * solver.dy2_inv

    du_dx = (jnp.roll(u_t, -1, axis=1) - jnp.roll(u_t, 1, axis=1)) / (2 * solver.dx)
    du_dy = (jnp.roll(u_t, -1, axis=2) - jnp.roll(u_t, 1, axis=2)) / (2 * solver.dy)

    # Evaluate model predictions on u_t grid per trajectory
    preds = jax.vmap(lambda u, dudx, dudy, d2udx, d2udy: vmap_PDE(solution, u, dudx, dudy, d2udx, d2udy, tree_evaluator))(u_t, du_dx, du_dy, d2u_dx2, d2u_dy2)
    # preds shape: (n_traj, n_time, nx, 2)
    f_pred = preds[..., 0]
    sigma_pred = preds[..., 1]

    var = (sigma_pred ** 2) * dt + 1e-5

    residual = (u_tp1 - u_t) - f_pred * dt
    nll = 0.5 * (jnp.log(2 * jnp.pi * var) + (residual ** 2) / var)
    total_nll = jnp.sum(nll)
    return total_nll

if __name__ == '__main__':
    # Generate synthetic 2D data
    # Initialize 2D solver
    nx, ny = 16, 16  # Grid points
    Lx, Ly = 4, 4 # Domain size
    dt = 0.001  # Time step
    batch_size = 2

    solver = SPDE_2D(nx, ny, Lx, Ly, dt)

    key = jr.PRNGKey(0)
    D = 0.1
    data_key, gp_key = jr.split(key)
    data = solver.generate_spde_data(data_key, D=D, T=1, n_trajectories=batch_size, save_every=20)
    test_data = solver.generate_spde_data(jr.PRNGKey(101), D=D, T=1, n_trajectories=4, save_every=20)
    _u, _x, _y, _t = test_data
    target_drift = jax.vmap(solver.drift, in_axes=(0, None))(_u[0], D)
    target_diffusion = jax.vmap(solver.noise_diffusion, in_axes=(None,None,0,0))(_x, _y, _t, _u[0])
    print(f"Data shape: {data[0].shape}")

    population_size = 100
    num_populations = 10
    optimize_constants_elite = 100
    max_nodes = 15
    num_generations = 50

    operator_list = [{"string": "+", "fn": lambda x, y: jnp.add(x, y), "arity": 2, "prob": 0.5},
                        {"string": "*", "fn": lambda x, y: jnp.multiply(x, y), "arity": 2, "prob": 0.5}
                         ]

    variable_list = [["u", "u_x", "u_y", "laplacian"]]

    fitness_function = FitnessFunctionSPDE_2D(solver)
    layer_sizes = jnp.array([2])

    strategy = GeneticProgramming(fitness_function=fitness_function, num_generations=num_generations, population_size=population_size, operator_list=operator_list, variable_list=variable_list, 
                                num_populations = num_populations, layer_sizes=layer_sizes, complexity_objective=True, constant_optimization=True, constant_optimization_steps=15, 
                                optimize_constants_elite=optimize_constants_elite, max_init_depth=5, constant_step_size=0.1, device_type="gpu", max_nodes=max_nodes, punish_duplicates=False)

    strategy.fit(gp_key, (data), verbose=0)

    u_all, x_all, y_all, t_all = data
    u_t  = u_all[:, :-1, :]   # shape (n_traj, time_len-1, nx)
    u_tp1 = u_all[:, 1:, :]

    # compute NLLs for whole pareto front (vectorized)
    pareto_fitness, pareto_front = strategy.pareto_front[0], strategy.pareto_front[1]
    best_solution = pareto_front[jnp.argmin(pareto_fitness)]

    drift_mses, diffusion_mses = validate(
            best_solution, _u[0], target_drift, target_diffusion, strategy.tree_evaluator
        )

    # Get the equation strings for drift and diffusion
    full_equation = strategy.expression_to_string(best_solution)

    print(f"equation = {full_equation}, drift MSE = {drift_mses}, diffusion MSE = {diffusion_mses}")

    # Store results for this target dimension
    result = {}
    result[f'u_equation'] = full_equation
    result[f'test_drift_mse'] = float(drift_mses)
    result[f'test_diffusion_mse'] = float(diffusion_mses)

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame([result])

    # Create filename based on experiment parameters
    filename = f"GP_SDE/SPDE_2D.csv"
    filepath = os.path.join("GP-SDE/data/", filename)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")