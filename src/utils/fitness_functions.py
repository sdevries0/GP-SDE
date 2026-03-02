
from typing import Tuple, Callable
import jax
import jax.numpy as jnp
from kozax.fitness_functions.base_fitness_function import BaseFitnessFunction

class FitnessFunctionSDE(BaseFitnessFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, candidate, data: Tuple, tree_evaluator: Callable) -> float:
        y, t, target_dim = data
        fitness = jax.vmap(self.predict, in_axes=[None, 0, 0, None, None])(candidate, y, t, tree_evaluator, jnp.squeeze(target_dim))
        return jnp.mean(fitness)

    def predict(self, candidate, ys, ts, tree_evaluator: Callable, target_dim: int):
        # Evaluate tree at current states
        tree_output = jax.vmap(lambda y: tree_evaluator(candidate, y))(ys)
        mu, sigma = tree_output[:, 0], tree_output[:, 1]
        
        shift_y = jnp.roll(ys, shift=-1, axis=0)
        dt_vals = (jnp.roll(ts, shift=-1)-ts)
        
        # Scale the drift and diffusion appropriately for the time scale
        var = sigma**2 * dt_vals + 1e-5 # variance scales with time interval

        pred_mu = ys[:, target_dim] + dt_vals * mu
        
        # Calculate negative log-likelihood
        fitness = -0.5 * ((shift_y[:, target_dim] - pred_mu) ** 2) / var - 0.5 * jnp.log(2*jnp.pi*var)
        
        # Aggregate results with weighted average
        total_fitness = -jnp.sum(fitness[:-1])
        
        return total_fitness
    
class FitnessFunctionODE(BaseFitnessFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, candidate, data: Tuple, tree_evaluator: Callable) -> float:
        y, t, target_dim = data
        fitness = jax.vmap(self.predict, in_axes=[None, 0, 0, None, None])(candidate, y, t, tree_evaluator, jnp.squeeze(target_dim))
        return jnp.mean(fitness)

    def predict(self, candidate, ys, ts, tree_evaluator: Callable, target_dim: int):
        # Evaluate tree at current states
        mu = jax.vmap(lambda y: tree_evaluator(candidate, y))(ys)
        
        # Create shifted arrays with the given scale
        shift_y = jnp.roll(ys, shift=-1, axis=0)
        dt_vals = (jnp.roll(ts, shift=-1)-ts)
        
        # Scale the drift and diffusion appropriately for the time scale  
        pred_mu = ys[:, target_dim] + dt_vals * mu[:,0]
        
        # Calculate negative log-likelihood
        fitness = ((shift_y[:, target_dim] - pred_mu) ** 2)
                            
        # Aggregate results with weighted average
        total_fitness = jnp.sum(fitness[:-1])
        
        return total_fitness
    
class FitnessFunctionSDEIntegration(BaseFitnessFunction):
    def __init__(self, n_substeps: int = 5, n_var = 1, score_region = None):
        super().__init__()
        self.n_substeps = n_substeps
        self.solve_steps = jnp.arange(self.n_substeps)
        self.n_var = n_var
        if score_region == None:
            self.score_region = jnp.arange(self.n_var)
        else:    
            self.score_region = score_region

    def __call__(self, candidate, data: Tuple, tree_evaluator: Callable) -> float:
        y, t = data
        fitness = jax.vmap(self.predict, in_axes=[None, 0, 0, None])(candidate, y, t, tree_evaluator)
        return jnp.mean(fitness)

    def predict(self, candidate, ys, ts, tree_evaluator: Callable):
        sub_dt = (ts[1] - ts[0])/self.n_substeps
        # Evaluate tree at current states
        def simulate_transition(carry, _):
            y_current = carry
            # Evaluate trees at current state - expecting 4 outputs: [drift_x1, drift_x2, diffusion_x1, diffusion_x2]
            tree_out = tree_evaluator(candidate, y_current)
            drift = tree_out[:self.n_var]  # First 2 outputs are drift components
            diffusion = tree_out[self.n_var:]  # Last 2 outputs are diffusion components
            
            # Take Euler step for both dimensions
            y_next = y_current.at[self.score_region].set(y_current[self.score_region] + drift * sub_dt)
            var_step = diffusion**2 * sub_dt
            
            return (y_next), var_step
        
        def compute_likelihood_single(y_start, y_end):
            # Simulate multiple substeps
            init_carry = (y_start)
            (y_pred), var_steps = jax.lax.scan(simulate_transition, init_carry, self.solve_steps)
            
            # Total variance accumulated over substeps for each dimension
            total_var = jnp.sum(var_steps, axis=0)  # Sum over substeps, keep dimensions
            
            # Calculate negative log-likelihood for both dimensions
            pred_error = y_end[self.score_region] - y_pred[self.score_region]
            nll = -0.5 * jnp.sum((pred_error**2) / total_var) - 0.5 * jnp.sum(jnp.log(2*jnp.pi*total_var))
            
            return -nll
        
        # Compute likelihood for all valid transitions
        shift_y = jnp.roll(ys, shift=-1, axis=0)
        likelihoods = jax.vmap(compute_likelihood_single)(ys[:-1], shift_y[:-1])
        
        total_fitness = jnp.sum(likelihoods)
        
        return total_fitness
    
class FitnessFunctionODEIntegration(BaseFitnessFunction):
    def __init__(self, n_substeps: int = 5):
        super().__init__()
        self.n_substeps = n_substeps
        self.solve_steps = jnp.arange(self.n_substeps)

    def __call__(self, candidate, data: Tuple, tree_evaluator: Callable) -> float:
        y, t = data
        fitness = jax.vmap(self.predict, in_axes=[None, 0, 0, None])(candidate, y, t, tree_evaluator)
        return jnp.mean(fitness)

    def predict(self, candidate, ys, ts, tree_evaluator: Callable):
        sub_dt = (ts[1] - ts[0])/self.n_substeps
        # Evaluate tree at current states
        def simulate_transition(carry, _):
            y_current = carry
            # Evaluate trees at current state
            tree_out = tree_evaluator(candidate, y_current)
            drift = tree_out  # First output is drift components
            
            # Take Euler step
            y_next = y_current + drift * sub_dt
            
            return (y_next), y_next
        
        def compute_MSE_single(y_start, y_end):
            # Simulate multiple substeps
            init_carry = (y_start)
            (y_pred), _ = jax.lax.scan(simulate_transition, init_carry, self.solve_steps)
                        
            # Calculate negative log-likelihood
            fitness = jnp.sum((y_end - y_pred) ** 2)
            
            return fitness
        
        # Compute likelihood for all valid transitions
        shift_y = jnp.roll(ys, shift=-1, axis=0)
        likelihoods = jax.vmap(compute_MSE_single)(ys[:-1], shift_y[:-1])
        
        total_fitness = jnp.sum(likelihoods)
        
        return total_fitness
    
class FitnessFunctionSPDE_1D(BaseFitnessFunction):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def __call__(self, candidate, data: Tuple, tree_evaluator: Callable) -> float:
        u, x, t = data  # 1D data: u(t,x), x coordinates, time coordinates
        fitness = jax.vmap(self.predict, in_axes=[None, 0, None, None])(candidate, u, t, tree_evaluator)
        return jnp.mean(fitness)

    def predict(self, candidate, us, ts, tree_evaluator: Callable):
        # Compute second spatial derivative (1D Laplacian) - FIXED: use axis=1 for space
        d2u_dx2 = (jnp.roll(us, 1, axis=1) - 2*us + jnp.roll(us, -1, axis=1)) * self.solver.dx2_inv
        du_dx = (jnp.roll(us, -1, axis=1) - jnp.roll(us, 1, axis=1)) / (2 * self.solver.dx)

        # Get next time step for prediction
        shift_u = jnp.roll(us, shift=-1, axis=0)
        dt_vals = (jnp.roll(ts, shift=-1)-ts)[:,None]  # 1D: only time and space dimensions

        tree_output = self.vmap(candidate, us, du_dx, d2u_dx2, tree_evaluator)
        mu, sigma = tree_output[..., 0], tree_output[..., 1]
        
        # Scale the drift and diffusion appropriately for the time scale
        var = sigma**2 * dt_vals + 1e-5 # variance scales with time interval

        pred_mu = us + dt_vals * mu
        
        # Calculate negative log-likelihood
        fitness = -0.5 * ((shift_u - pred_mu) ** 2) / var - 0.5 * jnp.log(2*jnp.pi*var)
        
        # Aggregate results with weighted average
        total_fitness = -jnp.sum(fitness[:-1])
        
        return total_fitness
    
    def vmap(self, candidate, us, du_dx, d2u_dx2, tree_evaluator):
        # 1D vectorization: only over space dimension
        vmap_x = jax.vmap(lambda u, dudx, d2udx: tree_evaluator(candidate, jnp.array([u, dudx, d2udx])))
        vmap_time = jax.vmap(lambda u, dudx, d2udx: vmap_x(u, dudx, d2udx), in_axes=[0,0,0])

        return vmap_time(us, du_dx, d2u_dx2)
    
class FitnessFunctionSPDE_2D(BaseFitnessFunction):
    def __init__(self, solver):
        super().__init__()
        self.solver = solver

    def __call__(self, candidate, data: Tuple, tree_evaluator: Callable) -> float:
        u, x, y, t = data  # 1D data: u(t,x), x coordinates, time coordinates
        fitness = jax.vmap(self.predict, in_axes=[None, 0, None, None])(candidate, u, t, tree_evaluator)
        return jnp.mean(fitness)

    def predict(self, candidate, us, ts, tree_evaluator: Callable):
        # Compute second spatial derivative (1D Laplacian) - FIXED: use axis=1 for space
        d2u_dx2 = (jnp.roll(us, 1, axis=1) - 2*us + jnp.roll(us, -1, axis=1)) * self.solver.dx2_inv
        d2u_dy2 = (jnp.roll(us, 1, axis=2) - 2*us + jnp.roll(us, -1, axis=2)) * self.solver.dy2_inv

        du_dx = (jnp.roll(us, -1, axis=1) - jnp.roll(us, 1, axis=1)) / (2 * self.solver.dx)
        du_dy = (jnp.roll(us, -1, axis=2) - jnp.roll(us, 1, axis=2)) / (2 * self.solver.dy)

        # Get next time step for prediction
        shift_u = jnp.roll(us, shift=-1, axis=0)
        dt_vals = (jnp.roll(ts, shift=-1)-ts)[:,None,None]  # 1D: only time and space dimensions

        tree_output = self.vmap(candidate, us, du_dx, du_dy, d2u_dx2, d2u_dy2, tree_evaluator)
        mu, sigma = tree_output[..., 0], tree_output[..., 1]
        
        # Scale the drift and diffusion appropriately for the time scale
        var = sigma**2 * dt_vals + 1e-5 # variance scales with time interval

        pred_mu = us + dt_vals * mu
        
        # Calculate negative log-likelihood
        fitness = -0.5 * ((shift_u - pred_mu) ** 2) / var - 0.5 * jnp.log(2*jnp.pi*var)
        
        # Aggregate results with weighted average
        total_fitness = -jnp.sum(fitness[:-1])
        
        return total_fitness
    
    def vmap(self, candidate, us, du_dx, du_dy, d2u_dx2, d2u_dy2, tree_evaluator):
        vmap_y = jax.vmap(lambda u, du_dx, du_dy, d2udx, d2udy: tree_evaluator(candidate, jnp.array([u, du_dx, du_dy, d2udx + d2udy])))
        vmap_x = jax.vmap(lambda u, du_dx, du_dy, d2udx, d2udy: vmap_y(u, du_dx, du_dy, d2udx, d2udy))
        vmap_time = jax.vmap(lambda u, du_dx, du_dy, d2udx, d2udy: vmap_x(u, du_dx, du_dy, d2udx, d2udy))

        return vmap_time(us, du_dx, du_dy, d2u_dx2, d2u_dy2)