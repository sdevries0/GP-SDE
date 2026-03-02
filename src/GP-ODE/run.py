import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import os
import time

from utils.SDE_envs import DoubleWell, LotkaVolterra, Lorenz96, VanDerPolOscillator, RosslerAttractor

from kozax.genetic_programming import GeneticProgramming
from utils.data_generator import generate_data
from utils.fitness_functions import FitnessFunctionODE
   
def validate(solution, grid, target_drift, tree_evaluator):
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
        
    # Compute MSE for drift
    drift_mse = jnp.mean((pred_drift - target_drift)**2)
        
    return drift_mse

if __name__ == '__main__':
    batch_size = 8

    env_name = sys.argv[1]
    if not os.path.exists(env_name):
        print(f"Environment '{env_name}' not recognized")
        sys.exit(1)

    population_size = 100
    num_populations = 5
    optimize_constants_elite = 100
    max_nodes = 15

    if env_name=="Double well":
        noise_level = 0.5
        diffusion_name = sys.argv[2] if len(sys.argv) > 2 else "additive"
        N_var = 1
        dt = 0.02
        env = DoubleWell(noise_level, diffusion_name)
        T = 50
        num_generations = 50

        save_path = f"DW_{diffusion_name}"

    elif env_name=="Lotka-Volterra":
        noise_level = 0.2
        env = LotkaVolterra(noise_level)
        T = 50
        num_generations = 50
        N_var = 2
        dt = jnp.float32(sys.argv[2]) if len(sys.argv) > 2 else 0.02

        save_path = f"LV_{dt}"

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

        save_path = f"standard/Lorenz_{N_var}"

    elif env_name=="Rossler":
        noise_level = 0.1
        env = RosslerAttractor(noise_level)
        T = 50
        N_var = 3
        dt = 0.02
        save_path = f"standard/Rossler"
        num_generations = 50

    elif env_name==vanderPol:
        noise_level = 0.2
        env = VanDerPolOscillator(noise_level)
        T = 50
        dt = 0.02
        N_var = 2
        save_path = f"standard/vdPol"
        num_generations = 50

    operator_list = [("+", lambda x, y: jnp.add(x, y), 2, 0.5), 
                    ("*", lambda x, y: jnp.multiply(x, y), 2, 0.5),
                    ]

    variable_list = [["x" + str(i) for i in range(env.n_var)]]

    fitness_function = FitnessFunctionODE()
    layer_sizes = jnp.array([1])

    strategy = GeneticProgramming(fitness_function=fitness_function, num_generations=num_generations, population_size=population_size, operator_list=operator_list, variable_list=variable_list, 
                                num_populations = num_populations, layer_sizes=layer_sizes, complexity_objective=True, constant_optimization_method="gradient", constant_optimization_steps=15, 
                                optimize_constants_elite=optimize_constants_elite, max_init_depth=5, constant_step_size_init=0.1, device_type="gpu", max_nodes=max_nodes)

    # Initialize list to collect results
    results = []
    times = []

    test_ts, test_ys = generate_data(jr.PRNGKey(101), env, 0.01, T, 16)
    test_grid = test_ys.reshape(test_ys.shape[0] * test_ys.shape[1], test_ys.shape[2])
    test_drift = jax.vmap(lambda x: env.drift(0, x, jnp.array([0])))(test_grid)

    for seed in range(10):
        key = jr.PRNGKey(seed)
        data_key, val_data_key, gp_key = jr.split(key, 3)
        ts, ys = generate_data(data_key, env, dt, T, batch_size)

        val_ts, val_ys = generate_data(val_data_key, env, dt, T, batch_size)

        val_grid = val_ys.reshape(val_ys.shape[0] * val_ys.shape[1], val_ys.shape[2])

        val_drift = jax.vmap(lambda x: env.drift(0, x, jnp.array([0])))(val_grid)

        # Initialize result dictionary for this seed
        seed_result = {
            'seed': seed,
        }
        
        N = 1 if env_name == "Lorenz96" else env.n_var
        
        for target_dim in range(10):

            strategy.fit(gp_key, (ys, ts, jnp.array([target_dim])), verbose=0)

            _val_drift = val_drift[:, target_dim]
            _test_drift = test_drift[:, target_dim]

            # Evaluate pareto front
            pareto_front = strategy.pareto_front[1]
            drift_mses = jax.vmap(validate, in_axes=[0,None,None,None])(
                pareto_front, val_grid, _val_drift, strategy.tree_evaluator
            )

            best_idx = jnp.argmin(drift_mses)
            best_solution = pareto_front[best_idx]

            test_drift_mse = validate(best_solution, test_grid, _test_drift, strategy.tree_evaluator)
            
            # Get the equation strings for drift and diffusion
            full_equation = strategy.expression_to_string(best_solution)
            
            # Store results for this target dimension
            seed_result[f'x{target_dim}_equation'] = full_equation
            seed_result[f'x{target_dim}_test_drift_mse'] = float(test_drift_mse)
            
            print(f"Seed {seed}, Target dim {target_dim}: equation = {full_equation}, drift MSE = {test_drift_mse}")
        
        # Add this seed's results to the main results list
        results.append(seed_result)

    # Convert results to DataFrame and save to CSV
    df = pd.DataFrame(results)

    # Create filename based on experiment parameters
    filename = f"data/GP_ODE/{save_path}.csv"

    # Save to CSV
    df.to_csv(filename, index=False)

    print(f"\nResults saved to: {filename}")
    print(f"Total experiments completed: {len(results)}")