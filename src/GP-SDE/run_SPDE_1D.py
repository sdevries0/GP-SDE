import sys
sys.path.insert(1, '../')
import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import os

from utils.SDE_envs import SPDE_1D
from kozax.genetic_programming import GeneticProgramming
from utils.fitness_functions import FitnessFunctionSPDE_1D
    
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
    du_dx = (jnp.roll(us, -1, axis=1) - jnp.roll(us, 1, axis=1)) / (2 * solver.dx)

    predictions = vmap_PDE(solution, us, du_dx, d2u_dx2, tree_evaluator)
    
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
    
def vmap_PDE(candidate, us, du_dx, d2u_dx2, tree_evaluator):
    vmap_x = jax.vmap(lambda u, dudx, d2udx: tree_evaluator(candidate, jnp.array([u, dudx, d2udx])))
    vmap_time = jax.vmap(lambda u, dudx, d2udx: vmap_x(u, dudx, d2udx), in_axes=[0,0,0])

    return vmap_time(us, du_dx, d2u_dx2)

if __name__ == '__main__':
    # Generate synthetic 1D data
    # Initialize 1D solver
    nx = 64  # Grid points for 1D
    Lx = 20.0  # Domain size
    dt = 0.001  # Time step
    batch_size = 4

    solver = SPDE_1D(nx, Lx, dt)

    key = jr.PRNGKey(0)
    data_key, gp_key = jr.split(key)
    data = solver.generate_spde_data(data_key, D=1.0, T=1, n_trajectories=batch_size, save_every=20)
    test_data = solver.generate_spde_data(jr.PRNGKey(101), D=1.0, T=1, n_trajectories=batch_size, save_every=20)
    _u, _x, _t = test_data
    target_drift = jax.vmap(solver.drift, in_axes=(0, None))(_u[0], 1.0)
    target_diffusion = jax.vmap(solver.noise_diffusion, in_axes=(None,0,0))(_x, _t, _u[0])
    print(f"Data shape: {data[0].shape}")

    population_size = 100
    num_populations = 10
    optimize_constants_elite = 100
    max_nodes = 15
    num_generations = 50

    operator_list = [("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
                    ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
                    ]

    variable_list = [["u", "u_x", "laplacian"]]

    fitness_function = FitnessFunctionSPDE_1D(solver)
    layer_sizes = jnp.array([2])

    strategy = GeneticProgramming(fitness_function=fitness_function, num_generations=num_generations, population_size=population_size, operator_list=operator_list, variable_list=variable_list, 
                                num_populations = num_populations, layer_sizes=layer_sizes, complexity_objective=True, constant_optimization_method="gradient", constant_optimization_steps=15, 
                                optimize_constants_elite=optimize_constants_elite, max_init_depth=5, constant_step_size_init=0.1, device_type="gpu", max_nodes=max_nodes)

    strategy.fit(gp_key, (data), verbose=5)

    # Evaluate pareto front
    pareto_front = strategy.pareto_front[1]
    drift_mses, diffusion_mses = jax.vmap(validate, in_axes=[0,None,None,None,None])(
        pareto_front, _u[0], target_drift, target_diffusion, strategy.tree_evaluator
    )

    best_idx = jnp.argmin(drift_mses + diffusion_mses)
    best_solution = pareto_front[best_idx]

    # Get the equation strings for drift and diffusion
    full_equation = strategy.expression_to_string(best_solution)

    print(f"equation = {full_equation}, drift MSE = {drift_mses[best_idx]}, diffusion MSE = {diffusion_mses[best_idx]}")

    # Store results for this target dimension
    result = {}
    result[f'u_equation'] = full_equation
    result[f'test_drift_mse'] = float(drift_mses[best_idx])
    result[f'test_diffusion_mse'] = float(diffusion_mses[best_idx])

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame([result])

    # Create filename based on experiment parameters
    filename = f"SDEs/GP_SDE/special/SPDE_1D.csv"
    filepath = os.path.join("/home/sdevries/results", filename)

    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to: {filepath}")