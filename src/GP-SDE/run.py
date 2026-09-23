import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import jax.random as jr

import os
import time

from utils.SDE_envs import DoubleWell, LotkaVolterra, VanDerPolOscillator, RosslerAttractor, Lorenz96
from utils.fitness_functions import FitnessFunctionSDE

from kozax.genetic_programming import GeneticProgramming
from utils.data_generator import generate_data
import pandas as pd
    
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
    pred_drift = predictions[:, 0]  # First output is drift
    pred_diffusion = jnp.abs(predictions[:, 1])  # Second output is diffusion (sigma)
    
    # Convert sigma to variance for comparison with target_diffusion
    pred_variance = pred_diffusion
    
    # Compute MSE for drift
    drift_mse = jnp.mean((pred_drift - target_drift)**2)
    
    # Compute MSE for diffusion
    diffusion_mse = jnp.mean((pred_variance - target_diffusion)**2)
    
    return drift_mse, diffusion_mse

def compute_nll_for_solution(solution, x_t_grid, dx_grid, dt, tree_evaluator, target_dim=0):
    """
    Compute negative log-likelihood of observed increments dx_grid under model defined by solution.

    Assumes univariate target_dim: dx ≈ f(x)*dt + sigma(x)*sqrt(dt)*ξ
    NLL per point: 0.5*(log(2π var) + residual^2/var)
    where var = sigma(x)^2 * dt (plus small floor to avoid zero)
    """
    # Evaluate model at states x_t_grid
    preds = jax.vmap(lambda x: tree_evaluator(solution, x))(x_t_grid)
    f_pred = preds[:, 0]
    sigma_pred = preds[:, 1]

    # Extract target-dimension components
    var = (sigma_pred ** 2) * dt + 1e-5

    residual = dx_grid[:, target_dim] - f_pred * dt
    nll = 0.5 * (jnp.log(2 * jnp.pi * var) + (residual ** 2) / var)
    # Sum over all points
    total_nll = jnp.sum(nll)
    return total_nll

if __name__ == '__main__':
    batch_size = 8

    env_name = sys.argv[1]

    population_size = 100
    num_populations = 5
    optimize_constants_elite = 100
    max_nodes = 15
    obs_noise = sys.argv[3] if len(sys.argv)>3 else 0.0

    if env_name=="Double well":
        noise_level = 0.5
        diffusion_name = sys.argv[2] if len(sys.argv) > 2 else "additive"
        N_var = 1
        dt = 0.02
        env = DoubleWell(noise_level, diffusion_name)
        T = 50
        num_generations = 50

        save_path = f"DW_{diffusion_name}"

        if float(obs_noise)>0.0:
            save_path = f"noise/DW_{obs_noise}"
        obs_noise = float(obs_noise)

    elif env_name=="Lotka-Volterra":
        noise_level = 0.2
        env = LotkaVolterra(noise_level)
        T = 50
        num_generations = 50
        N_var = 2
        dt = sys.argv[2] if len(sys.argv) > 2 else 0.02

        save_path = f"LV_{dt}"

        dt = jnp.float32(dt)

    elif env_name=="Lorenz96":
        noise_level = 0.2
        N_var = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        dt = 0.02
        env = Lorenz96(N_var, noise_level, 4)
        T = 25
        if N_var == 5:
            num_populations = 10
            num_generations = 100
        else:
            num_populations = 20
        
            num_generations = 200
        optimize_constants_elite = 200
        max_nodes = 20

        save_path = f"Lorenz_{N_var}"

    elif env_name=="Rossler":
        noise_level = 0.1
        env = RosslerAttractor(noise_level)
        T = 50
        N_var = 3
        dt = 0.02
        save_path = f"Rossler"
        num_generations = 50

    elif env_name=="vanderPol":
        noise_level = 0.2
        env = VanDerPolOscillator(noise_level)
        T = 50
        dt = 0.02
        N_var = 2
        save_path = f"vdPol"
        num_generations = 50


    operator_list = [{"string": "+", "fn": lambda x, y: jnp.add(x, y), "arity": 2, "prob": 0.5},
                    {"string": "*", "fn": lambda x, y: jnp.multiply(x, y), "arity": 2, "prob": 0.5}
                     ]

    variable_list = [["x" + str(i) for i in range(env.n_var)]]

    fitness_function = FitnessFunctionSDE()
    layer_sizes = jnp.array([2])

    strategy = GeneticProgramming(fitness_function=fitness_function, num_generations=num_generations, population_size=population_size, operator_list=operator_list, variable_list=variable_list, 
                                num_populations = num_populations, layer_sizes=layer_sizes, complexity_objective=True, constant_optimization=True, constant_optimization_steps=15, 
                                optimize_constants_elite=optimize_constants_elite, max_init_depth=5, constant_step_size=0.1, device_type="gpu", max_nodes=max_nodes, punish_duplicates=False)

    # Initialize list to collect results
    results = []
    times = []

    test_ts, test_ys = generate_data(jr.PRNGKey(101), env, 0.01, T, 16)
    test_grid = test_ys.reshape(test_ys.shape[0] * test_ys.shape[1], test_ys.shape[2])
    test_drift = jax.vmap(lambda x: env.drift(0, x, jnp.array([0])))(test_grid)
    test_diffusion = jax.vmap(lambda x: env.diffusion(0, x, jnp.array([0])))(test_grid)

    for seed in range(10):
        key = jr.PRNGKey(seed)
        data_key, gp_key = jr.split(key, 2)
        ts, ys = generate_data(data_key, env, dt, T, batch_size, obs_sigma=obs_noise)

        # Initialize result dictionary for this seed
        seed_result = {
            'seed': seed,
        }

        N = 1 if env_name == "Lorenz96" else env.n_var
        
        for target_dim in range(N):
            strategy.fit(gp_key, (ys, ts, jnp.array([target_dim])), verbose=0)

            _test_drift = test_drift[:, target_dim]
            _test_diffusion = test_diffusion[:, target_dim, target_dim]

            # Evaluate pareto front
            pareto_front = strategy.pareto_front[1]

            # Compute complexity for each pareto solution (node count)
            complexities = jax.vmap(lambda s: jnp.sum(s[:,:,0] != 0))(pareto_front)

            # Prepare training increments: x_t and dx = x_{t+1}-x_t
            x_t = ys[:, :-1, :]
            x_tp1 = ys[:, 1:, :]
            dx = x_tp1 - x_t
            x_t_grid_inc = x_t.reshape(x_t.shape[0] * x_t.shape[1], x_t.shape[2])
            dx_grid_inc = dx.reshape(dx.shape[0] * dx.shape[1], dx.shape[2])

            # Compute NLL for each pareto solution on training increments
            nlls = jax.vmap(lambda s: compute_nll_for_solution(s, x_t_grid_inc, dx_grid_inc, dt, strategy.tree_evaluator, target_dim))(
                pareto_front
            )

            # Define MDL as model complexity + negative log-likelihood (total)
            mdl_scores = complexities * jnp.log(len(strategy.node_function_list)-1) + nlls

            best_idx = jnp.argmin(mdl_scores)
            best_solution = pareto_front[best_idx]

            test_drift_mse, test_diffusion_mse = validate(best_solution, test_grid, _test_drift, _test_diffusion, strategy.tree_evaluator)
            
            # Get the equation strings for drift and diffusion
            full_equation = strategy.expression_to_string(best_solution)
            
            # Store results for this target dimension
            seed_result[f'x{target_dim}_equation'] = full_equation
            seed_result[f'x{target_dim}_test_drift_mse'] = float(test_drift_mse)
            seed_result[f'x{target_dim}_test_diffusion_mse'] = float(test_diffusion_mse)
            
            print(f"Seed {seed}, Target dim {target_dim}: equation = {full_equation}, drift MSE = {test_drift_mse}, diffusion MSE = {test_diffusion_mse}")

        # Add this seed's results to the main results list
        results.append(seed_result)

    df = pd.DataFrame(results)

    # Create filename based on experiment parameters
    filename = f"GP-SDE/data/GP_SDE/{save_path}.csv"

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"\nResults saved to: {filename}")
    print(f"Total experiments completed: {len(results)}")