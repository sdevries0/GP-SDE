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


def compute_nll_for_solution(solution, x_t_grid, dx_grid, dt, tree_evaluator, n_substeps=1):
    """
    Compute negative log-likelihood of observed increments dx_grid under model defined by solution,
    accounting for `n_substeps` sub-integration steps used by the fitness function.

    We simulate the deterministic mean increment by repeatedly evaluating the drift and taking
    Euler steps of size dt_sub = dt / n_substeps. The total variance is the sum of per-substep
    variances: sum(sigma(x_k)^2 * dt_sub).
    """
    sub_dt = dt / n_substeps

    dim = x_t_grid.shape[1]

    # prepare scan indices
    steps = jnp.arange(int(n_substeps))

    def simulate_transition(y_current, _):
        tree_out = tree_evaluator(solution, y_current)
        drift = tree_out[:dim]
        diffusion = tree_out[dim:dim*2]
        y_next = y_current + drift * sub_dt
        var_step = diffusion**2 * sub_dt
        return y_next, var_step

    def compute_nll_single(y_start, dx_row):
        y_end = y_start + dx_row
        y_pred, var_steps = jax.lax.scan(simulate_transition, y_start, steps)
        # var_steps shape (n_substeps, dim)
        total_var = jnp.sum(var_steps, axis=0) + 1e-5

        pred_error = y_end - y_pred
        # sum over dimensions to get scalar per-sample
        nll = 0.5 * (jnp.sum(jnp.log(2 * jnp.pi * total_var)) + jnp.sum((pred_error**2) / total_var))
        return nll

    # Vectorize across all training transitions
    nlls = jax.vmap(compute_nll_single)(x_t_grid, dx_grid)
    total_nll = jnp.sum(nlls)
    return total_nll

if __name__ == '__main__':
    batch_size = 8

    env_name = sys.argv[1]
    dt = sys.argv[2] if len(sys.argv) > 2 else 0.2

    T = 50
    noise_level = 0.2

    env = LotkaVolterra(noise_level)

    population_size = 200
    num_populations = 10 + 5 * (jnp.float32(dt)==0.5)
    num_generations = 100

    operator_list = [{"string": "+", "fn": lambda x, y: jnp.add(x, y), "arity": 2, "prob": 0.5},
                {"string": "*", "fn": lambda x, y: jnp.multiply(x, y), "arity": 2, "prob": 0.5}
                 ]

    variable_list = [["x" + str(i) for i in range(env.n_var)]]

    n_substeps = 5

    fitness_function = FitnessFunctionSDEIntegration(n_substeps=n_substeps, n_var = env.n_var)
    layer_sizes = jnp.array([2*env.n_var])

    strategy = GeneticProgramming(fitness_function=fitness_function, num_generations=num_generations, population_size=population_size, operator_list=operator_list, variable_list=variable_list, 
                                num_populations = num_populations, layer_sizes=layer_sizes, complexity_objective=True, constant_optimization=True, constant_optimization_steps=50, 
                                optimize_constants_elite=500, max_init_depth=5, constant_step_size=0.1, device_type="gpu", reproduction_probability_factors=(0.2, 1.0), punish_duplicates=False)

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
        data_key, gp_key = jr.split(key)
        ts, ys = generate_data(data_key, env, jnp.float32(dt), T, batch_size)

        strategy.fit(gp_key, (ys, ts), verbose=0)

        # Evaluate pareto front and compute MDL = complexity + NLL on training increments
        pareto_front = strategy.pareto_front[1]

        # Compute complexity (node-count) for each candidate
        complexities = jax.vmap(lambda s: jnp.sum(s[:,:,0] != 0))(pareto_front)

        # Prepare training increments: x_t and dx = x_{t+1}-x_t
        x_t = ys[:, :-1, :]
        x_tp1 = ys[:, 1:, :]
        dx = x_tp1 - x_t
        x_t_grid_inc = x_t.reshape(x_t.shape[0] * x_t.shape[1], x_t.shape[2])
        dx_grid_inc = dx.reshape(dx.shape[0] * dx.shape[1], dx.shape[2])

        # Compute NLL for each pareto solution on training increments
        nlls = jax.vmap(lambda s: compute_nll_for_solution(s, x_t_grid_inc, dx_grid_inc, jnp.float32(dt), strategy.tree_evaluator, n_substeps))(
            pareto_front
        )

        # Define MDL as model complexity + negative log-likelihood (total)
        mdl_scores = complexities * jnp.log(len(strategy.node_function_list)-1) + nlls

        best_idx = int(jnp.argmin(mdl_scores))
        best_solution = pareto_front[best_idx]
        full_equation = strategy.expression_to_string(best_solution)

        # For backwards-compatibility store drift/diffusion MSEs on test grid as well
        test_drift_mse, test_diffusion_mse = validate(best_solution, test_grid, test_drift, test_diffusion, strategy.tree_evaluator)
        print(f"Selected by MDL: equation = {full_equation}, drift MSE = {test_drift_mse}, diffusion MSE = {test_diffusion_mse}")

        seed_result[f'equations'] = full_equation
        seed_result[f'test_drift_mse'] = float(test_drift_mse)
        seed_result[f'test_diffusion_mse'] = float(test_diffusion_mse)

        results.append(seed_result)

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Create filename based on experiment parameters
    filename = f"GP-SDE/data/GP_SDE/LV_MS_{dt}.csv"

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"\nResults saved to: {filename}")
    print(f"Total experiments completed: {len(results)}")