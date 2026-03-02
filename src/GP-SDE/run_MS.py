import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import os

from utils.SDE_envs import LotkaVolterra

from kozax.genetic_programming import GeneticProgramming
from utils.data_generator import generate_data
from utils.fitness_functions import FitnessFunctionSDEIntegration
    
def validate(solution, grid, target_drift, target_diffusion, tree_evaluator):
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
    predictions = jax.vmap(lambda x: tree_evaluator(solution, x))(grid)
    
    # Extract drift and diffusion predictions
    pred_drift = predictions[:, :env.n_var]  # First output is drift
    pred_diffusion = predictions[:, env.n_var:]  # Second output is diffusion (sigma)
    
    # Convert sigma to variance for comparison with target_diffusion
    pred_variance = jnp.abs(pred_diffusion)
    
    # Compute MSE for drift
    drift_mse = jnp.mean((pred_drift - target_drift)**2)
    
    # Compute MSE for diffusion
    diffusion_mse = jnp.mean((pred_variance - target_diffusion)**2)
    
    return drift_mse, diffusion_mse

if __name__ == '__main__':
    batch_size = 8

    env_name = sys.argv[1]
    dt = jnp.float32(sys.argv[2]) if len(sys.argv) > 2 else 0.2

    T = 50
    noise_level = 0.2

    env = LotkaVolterra(noise_level)

    population_size = 200
    num_populations = 10 + 5 * (dt==0.5)
    num_generations = 100

    operator_list = [("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
                    ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
                    ]

    variable_list = [["x" + str(i) for i in range(env.n_var)]]

    n_substeps = 5

    fitness_function = FitnessFunctionSDEIntegration(n_substeps=n_substeps, n_var = env.n_var)
    layer_sizes = jnp.array([2*env.n_var])

    strategy = GeneticProgramming(fitness_function=fitness_function, num_generations=num_generations, population_size=population_size, operator_list=operator_list, variable_list=variable_list, 
                                num_populations = num_populations, layer_sizes=layer_sizes, complexity_objective=True, constant_optimization_method="gradient", constant_optimization_steps=50, 
                                optimize_constants_elite=500, max_init_depth=5, constant_step_size_init=0.1, device_type="gpu", reproduction_probability_factors=(0.2, 1.0))

    results = []

    test_ts, test_ys = generate_data(jr.PRNGKey(101), env, 0.01, T, 16)
    test_grid = test_ys.reshape(test_ys.shape[0] * test_ys.shape[1], test_ys.shape[2])
    test_drift = jax.vmap(lambda x: env.drift(0, x, jnp.array([0])))(test_grid)
    test_diffusion = jax.vmap(lambda x: jnp.diag(env.diffusion(0, x, jnp.array([0]))))(test_grid)

    for seed in range(10):
        seed_result = {
            'seed': seed,
        }

        key = jr.PRNGKey(seed)
        data_key, val_data_key, gp_key = jr.split(key, 3)
        ts, ys = generate_data(data_key, env, dt, T, batch_size)

        strategy.fit(gp_key, (ys, ts), verbose=10)

        val_ts, val_ys = generate_data(val_data_key, env, dt, T, batch_size)

        val_grid = val_ys.reshape(val_ys.shape[0] * val_ys.shape[1], val_ys.shape[2])

        val_drift = jax.vmap(lambda x: env.drift(0, x, jnp.array([0])))(val_grid)
        val_diffusion = jax.vmap(lambda x: jnp.diag(env.diffusion(0, x, jnp.array([0]))))(val_grid)

        # Evaluate pareto front
        pareto_front = strategy.pareto_front[1]
        drift_mses, diffusion_mses = jax.vmap(validate, in_axes=[0,None,None,None,None])(
            pareto_front, val_grid, val_drift, val_diffusion, strategy.tree_evaluator
        )

        best_idx = jnp.argmin(drift_mses + diffusion_mses)
        best_solution = pareto_front[best_idx]
        full_equation = strategy.expression_to_string(best_solution)

        test_drift_mse, test_diffusion_mse = validate(pareto_front[best_idx], test_grid, test_drift, test_diffusion, strategy.tree_evaluator)
        print(f"Best diffusion (validation): equation = {full_equation}, drift MSE = {test_drift_mse}, diffusion MSE = {test_diffusion_mse}")

        seed_result[f'equations'] = full_equation
        seed_result[f'test_drift_mse'] = float(test_drift_mse)
        seed_result[f'test_diffusion_mse'] = float(test_diffusion_mse)

        results.append(seed_result)

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Create filename based on experiment parameters
    filename = f"data/GP_SDE/LV_MS_{dt}.csv"

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"\nResults saved to: {filename}")
    print(f"Total experiments completed: {len(results)}")