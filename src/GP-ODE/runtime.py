from functools import partial
import sys
sys.path.insert(1, '../')
import jax
import jax.numpy as jnp
import jax.random as jr
import pandas as pd
import os
import time

from utils.SDE_envs import Lorenz96
from utils.fitness_functions import FitnessFunctionODE

from utils.my_GP import GeneticProgramming
from utils.data_generator import generate_data
    
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

    N_var = int(sys.argv[1])
    dt = 0.02
    ts_type = "normal"

    population_size = 100
    max_nodes = 15

    noise_level = 0.2
    env = Lorenz96(N_var, noise_level, 4)
    T = 25
    num_populations = 5
    num_generations = 50

    optimize_constants_elite = 100


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
    times = []

    for seed in range(11):
        key = jr.PRNGKey(seed)
        data_key, val_data_key, gp_key = jr.split(key, 3)
        ts, ys = generate_data(data_key, env, dt, T, batch_size, ts_type = ts_type)

        val_ts, val_ys = generate_data(val_data_key, env, dt, T, batch_size, ts_type = ts_type)

        val_grid = val_ys.reshape(val_ys.shape[0] * val_ys.shape[1], val_ys.shape[2])

        val_drift = jax.vmap(lambda x: env.drift(0, x, jnp.array([0])))(val_grid)

        # Initialize result dictionary for this seed
        seed_result = {
            'seed': seed,
        }

        N = 1
        
        for target_dim in range(N):
            start = time.time()
            strategy.fit(gp_key, (ys, ts, jnp.array([target_dim])), verbose=0)
            end = time.time()
            print(end-start)
        
            # Add this seed's results to the main results list
            times.append(end-start)

    # Make sure the results directory exists
    jnp.save(os.path.join("/home/sdevries/results", f"SDEs/runtimes/GP_ODE_{N_var}") + "_time.npy", jnp.array(times))