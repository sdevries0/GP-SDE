"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 sdevries0

This work is licensed under the Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from kozax.environments.SR_environments.time_series_environment_base import EnvironmentBase
from jaxtyping import Array
from typing import Tuple, Dict

class LotkaVolterra(EnvironmentBase):
    """
    Lotka-Volterra environment for symbolic regression tasks.

    Parameters
    ----------
    process_noise : float, optional
        Standard deviation of the process noise. Default is 0.

    Attributes
    ----------
    init_mu : :class:`jax.Array`
        Mean of the initial state distribution.
    init_sd : float
        Standard deviation of the initial state distribution.
    alpha : float
        Parameter alpha of the Lotka-Volterra system.
    beta : float
        Parameter beta of the Lotka-Volterra system.
    delta : float
        Parameter delta of the Lotka-Volterra system.
    gamma : float
        Parameter gamma of the Lotka-Volterra system.
    V : :class:`jax.Array`
        Process noise covariance matrix.

    Methods
    -------
    sample_init_states(batch_size, key)
        Samples initial states for the environment.
    drift(t, state, args)
        Computes the drift function for the environment.
    diffusion(t, state, args)
        Computes the diffusion function for the environment.
    """

    def __init__(self, process_noise: float = 0) -> None:
        n_var = 2
        super().__init__(n_var, process_noise)

        self.alpha = 1.1
        self.beta = 0.4
        self.delta = 0.1
        self.gamma = 0.4
        self.V = self.process_noise * jnp.eye(self.n_var)

    def sample_init_states(self, batch_size: int, key: jr.PRNGKey) -> Array:
        """
        Samples initial states for the environment.

        Parameters
        ----------
        batch_size : int
            Number of initial states to sample.
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.

        Returns
        -------
        :class:`jax.Array`
            Initial states.
        """
        return jr.uniform(key, shape=(batch_size, self.n_var), minval=5, maxval=10)
    
    def drift(self, t: float, state: Array, args: float = jnp.array([0.0])) -> Array:
        """
        Computes the drift function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Drift.
        """
        x, y = state
        return jnp.array([self.alpha * x - self.beta * x * y, 
                          self.delta * x * y - self.gamma * y])

    def diffusion(self, t: float, state: Array, args: Tuple) -> Array:
        """
        Computes the diffusion function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Diffusion.
        """
        return self.V * state

class DoubleWell(EnvironmentBase):
    def __init__(self, process_noise: float = 0, diffusion_type = "nonlinear") -> None:
        n_var = 1
        super().__init__(n_var, process_noise)

        self.init_mu = jnp.array([0.0])
        self.init_sd = jnp.array([0.1])

        self.V = self.process_noise * jnp.eye(self.n_var)

        if diffusion_type == "nonlinear":
            self._diffusion = lambda t, x, args: self.V*(1 + x**2)
        elif diffusion_type == "linear":
            self._diffusion = lambda t, x, args: self.V*jnp.abs(x)
        elif diffusion_type == "additive":
            self._diffusion = lambda t, x, args: self.V

    def sample_init_states(self, batch_size: int, key: jr.PRNGKey) -> Array:
        return self.init_mu + self.init_sd * jr.normal(key, shape=(batch_size, 1))
    
    def drift(self, t: float, state: Array, args: Tuple) -> Array:
        x = state[0]
        return jnp.array([x - x**3])
    
    def diffusion(self, t: float, state: Array, args: Tuple) -> Array:
        return self._diffusion(t, state, args)
    
class VanDerPolOscillator(EnvironmentBase):
    def __init__(self, process_noise: float = 0) -> None:
        n_var = 2
        super().__init__(n_var, process_noise)

        self.init_mu = jnp.array([0, 0])
        self.init_sd = jnp.array([1.0, 1.0])

        self.mu = 1
        self.V = self.process_noise * jnp.eye(self.n_var)

    def sample_init_states(self, batch_size: int, key: jr.PRNGKey) -> Array:
        return self.init_mu + self.init_sd * jr.normal(key, shape=(batch_size, self.n_var))
    
    def drift(self, t: float, state: Array, args: Tuple) -> Array:
        return jnp.array([state[1], self.mu * (1 - state[0]**2) * state[1] - state[0]])

    def diffusion(self, t: float, state: Array, args: Tuple) -> Array:
        return self.V*jnp.array([0., 1 + 0.5 * state[0]**2])

class LorenzAttractor(EnvironmentBase):
    """
    Lorenz Attractor environment for symbolic regression tasks.

    Parameters
    ----------
    process_noise : float, optional
        Standard deviation of the process noise. Default is 0.

    Attributes
    ----------
    init_mu : :class:`jax.Array`
        Mean of the initial state distribution.
    init_sd : float
        Standard deviation of the initial state distribution.
    sigma : float
        Parameter sigma of the Lorenz system.
    rho : float
        Parameter rho of the Lorenz system.
    beta : float
        Parameter beta of the Lorenz system.
    V : :class:`jax.Array`
        Process noise covariance matrix.

    Methods
    -------
    sample_init_states(batch_size, key)
        Samples initial states for the environment.
    drift(t, state, args)
        Computes the drift function for the environment.
    diffusion(t, state, args)
        Computes the diffusion function for the environment.
    """

    def __init__(self, process_noise: float = 0) -> None:
        n_var = 3
        super().__init__(n_var, process_noise)

        self.init_mu = jnp.array([1, 1, 1])
        self.init_sd = 1.0

        self.sigma = 10
        self.rho = 28
        self.beta = 8 / 3
        self.V = self.process_noise * jnp.eye(self.n_var)

    def sample_init_states(self, batch_size: int, key: jr.PRNGKey) -> Array:
        """
        Samples initial states for the environment.

        Parameters
        ----------
        batch_size : int
            Number of initial states to sample.
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.

        Returns
        -------
        :class:`jax.Array`
            Initial states.
        """
        return self.init_mu + self.init_sd * jr.normal(key, shape=(batch_size, self.n_var))

    def drift(self, t: float, state: Array, args: Tuple = jnp.array([0.0])) -> Array:
        """
        Computes the drift function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Drift.
        """
        return jnp.array([self.sigma * (state[1] - state[0]), state[0] * (self.rho - state[2]) - state[1], state[0] * state[1] - self.beta * state[2]])

    def diffusion(self, t: float, state: Array, args: Tuple) -> Array:
        """
        Computes the diffusion function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Diffusion.
        """
        return self.V
    
class RosslerAttractor(EnvironmentBase):
    """
    Rossler Attractor environment for symbolic regression tasks.

    Parameters
    ----------
    process_noise : float, optional
        Standard deviation of the process noise. Default is 0.

    Attributes
    ----------
    init_mu : :class:`jax.Array`
        Mean of the initial state distribution.
    init_sd : float
        Standard deviation of the initial state distribution.
    sigma : float
        Parameter sigma of the Lorenz system.
    rho : float
        Parameter rho of the Lorenz system.
    beta : float
        Parameter beta of the Lorenz system.
    V : :class:`jax.Array`
        Process noise covariance matrix.

    Methods
    -------
    sample_init_states(batch_size, key)
        Samples initial states for the environment.
    drift(t, state, args)
        Computes the drift function for the environment.
    diffusion(t, state, args)
        Computes the diffusion function for the environment.
    """

    def __init__(self, process_noise: float = 0) -> None:
        n_var = 3
        super().__init__(n_var, process_noise)

        self.init_mu = jnp.array([1, 1, 1])
        self.init_sd = 1.0

        self.a = 0.2
        self.b = 0.2
        self.c = 5.7
        self.V = self.process_noise * jnp.eye(self.n_var)

    def sample_init_states(self, batch_size: int, key: jr.PRNGKey) -> Array:
        """
        Samples initial states for the environment.

        Parameters
        ----------
        batch_size : int
            Number of initial states to sample.
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.

        Returns
        -------
        :class:`jax.Array`
            Initial states.
        """
        return self.init_mu + self.init_sd * jr.normal(key, shape=(batch_size, self.n_var))

    def drift(self, t: float, state: Array, args: Tuple = jnp.array([0.0])) -> Array:
        """
        Computes the drift function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Drift.
        """
        return jnp.array([-(state[1] + state[2]), state[0] + self.a*state[1], self.b + state[2] * (state[0] - self.c)])

    def diffusion(self, t: float, state: Array, args: Tuple) -> Array:
        """
        Computes the diffusion function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Diffusion.
        """
        return self.V * jnp.abs(state)
    
class Lorenz96(EnvironmentBase):
    """
    Lorenz 96 environment for symbolic regression tasks.

    Parameters
    ----------
    n_dim : int, optional
        Number of dimensions. Default is 3.
    process_noise : float, optional
        Standard deviation of the process noise. Default is 0.

    Attributes
    ----------
    F : float
        Forcing term.
    init_mu : :class:`jax.Array`
        Mean of the initial state distribution.
    init_sd : float
        Standard deviation of the initial state distribution.
    V : :class:`jax.Array`
        Process noise covariance matrix.

    Methods
    -------
    sample_init_states(batch_size, key)
        Samples initial states for the environment.
    drift(t, state, args)
        Computes the drift function for the environment.
    diffusion(t, state, args)
        Computes the diffusion function for the environment.
    """

    def __init__(self, n_dim: int = 3, process_noise: float = 0, F: float = 8) -> None:
        n_var = n_dim
        super().__init__(n_var, process_noise)

        self.F = F
        self.init_mu = jnp.ones(self.n_var) * self.F
        self.init_sd = 0.1

        self.V = self.process_noise * jnp.eye(self.n_var)

    def sample_init_states(self, batch_size: int, key: jr.PRNGKey) -> Array:
        """
        Samples initial states for the environment.

        Parameters
        ----------
        batch_size : int
            Number of initial states to sample.
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.

        Returns
        -------
        :class:`jax.Array`
            Initial states.
        """
        return self.init_mu + self.init_sd * jr.normal(key, shape=(batch_size, self.n_var))

    def drift(self, t: float, state: Array, args: Tuple) -> Array:
        """
        Computes the drift function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Drift.
        """
        f = lambda x_cur, x_next, x_prev1, x_prev2: (x_next - x_prev2) * x_prev1 - x_cur + self.F
        return jax.vmap(f)(state, jnp.roll(state, -1), jnp.roll(state, 1), jnp.roll(state, 2))

    def diffusion(self, t: float, state: Array, args: Tuple) -> Array:
        """
        Computes the diffusion function for the environment.

        Parameters
        ----------
        t : float
            Current time.
        state : :class:`jax.Array`
            Current state.
        args : tuple
            Additional arguments.

        Returns
        -------
        :class:`jax.Array`
            Diffusion.
        """
        return self.V * jnp.abs(state)
    
class SPDE_2D:
    """JAX-based 1D SPDE solver using finite differences"""
    
    def __init__(self, nx: int, ny: int, Lx: float, Ly: float, dt: float, true_system: bool):
        self.nx, self.ny = nx, ny
        self.Lx, self.Ly = Lx, Ly
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.dt = dt
        self.true_system = true_system
        
        # Create spatial grids
        self.x = jnp.linspace(0, Lx, nx)
        self.y = jnp.linspace(0, Ly, ny)
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing='ij')
        
        # Precompute finite difference coefficients
        self.dx2_inv = 1.0 / (self.dx ** 2)
        self.dy2_inv = 1.0 / (self.dy ** 2)
    
    def laplacian(self, u: jnp.ndarray) -> jnp.ndarray:
        """Compute 1D Laplacian using finite differences with periodic BC"""
        d2u_dx2 = (jnp.roll(u, 1, axis=0) - 2*u + jnp.roll(u, -1, axis=0)) * self.dx2_inv
        d2u_dy2 = (jnp.roll(u, 1, axis=1) - 2*u + jnp.roll(u, -1, axis=1)) * self.dy2_inv
        return d2u_dx2 + d2u_dy2
    
    def gradient(self, u: jnp.ndarray) -> jnp.ndarray:
        """Compute 1D gradient using central differences"""
        du_dx = (jnp.roll(u, -1, axis=0) - jnp.roll(u, 1, axis=0)) / (2 * self.dx)
        du_dy = (jnp.roll(u, -1, axis=1) - jnp.roll(u, 1, axis=1)) / (2 * self.dy)
        return du_dx, du_dy
    
    def generate_noise(self, key: jnp.ndarray, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Generate space-time correlated noise"""
        # Generate white noise
        noise = jr.normal(key, shape)
        
        # Apply spatial smoothing for more realistic noise
        # Simple box filter
        kernel = jnp.array([[0.,0.,0.,],[0.,1.,0.,],[0.,0.,0.,]])
        
        # Pad for convolution
        noise_padded = jnp.pad(noise, ((1, 1), (1, 1)), mode='wrap')
        
        # Apply convolution manually
        smoothed_noise = jnp.zeros_like(noise)
        for i in range(3):
            for j in range(3):
                smoothed_noise += kernel[i, j] * noise_padded[i:i+self.nx, j:j+self.ny]
        
        return smoothed_noise
    
    def drift(self, u, D):
        """True reaction term to be discovered
        
        Examples:
        - Fisher-KPP: r*u*(1-u/K)
        """
        # Fisher-KPP type reaction: u*(1-u)
        if self.true_system:
            laplacian_u = self.laplacian(u)
            diffusion = D * laplacian_u
        else:
            laplacian_u = 0.0979*self.laplacian(u)
            diffusion = laplacian_u
        return diffusion

    def noise_diffusion(self, x, y, t, u):
        """True noise coefficient function to be discovered"""
        dudx, dudy = self.gradient(u)
        return (dudx**2 + dudy**2)

    def generate_spde_data(
        self,
        key,
        D: float = 0.1,
        T: float = 10.0,
        n_trajectories: int = 3,
        save_every: int = 20,
    ) -> Dict[str, jnp.ndarray]:
        """Generate synthetic 1D reaction-diffusion data (vectorized over trajectories)."""

        n_steps = int(T / self.dt)
        save_idx = jnp.arange(0, n_steps, save_every)
        n_save = save_idx.shape[0]

        traj_keys = jr.split(key, n_trajectories)

        def simulate_one(traj_key):
            u0 = jnp.sin(0.5*jnp.pi*self.X) * jnp.cos(0.5*jnp.pi*self.Y) * 0.2

            def step_fn(carry, step):
                key_i, u = carry
                key_i, noise_key = jr.split(key_i)

                t = step * self.dt
                deterministic = self.drift(u, D)
                noise = self.generate_noise(noise_key, (self.nx,self.ny))
                noise_coeff = self.noise_diffusion(self.x, self.y, t, u)
                stochastic = noise_coeff * noise * jnp.sqrt(self.dt)

                u_next = u + self.dt * deterministic + stochastic
                return (key_i, u_next), u_next

            (_, _), u_hist = jax.lax.scan(step_fn, (traj_key, u0), jnp.arange(n_steps))
            return u_hist[save_idx]  # (n_save, nx)

        all_u = jax.vmap(simulate_one)(traj_keys)  # (n_trajectories, n_save, nx)
        all_times = save_idx * self.dt

        return all_u, self.x, self.y, all_times
    
class SPDE_1D:
    """JAX-based 1D SPDE solver using finite differences"""
    
    def __init__(self, nx: int, Lx: float, dt: float, true_system: bool):
        self.nx = nx
        self.Lx = Lx
        self.dx = Lx / nx  # Periodic domain
        self.dt = dt
        self.true_system = true_system
        
        # Create spatial grid
        self.x = jnp.linspace(0, Lx, nx, endpoint=False)  # Periodic
        
        # Precompute finite difference coefficients
        self.dx2_inv = 1.0 / (self.dx ** 2)
    
    def laplacian(self, u: jnp.ndarray) -> jnp.ndarray:
        """Compute 1D Laplacian using finite differences with periodic BC"""
        d2u_dx2 = (jnp.roll(u, 1, axis=0) - 2*u + jnp.roll(u, -1, axis=0)) * self.dx2_inv
        return d2u_dx2
    
    def generate_noise(self, key: jnp.ndarray, shape: Tuple[int, ...]) -> jnp.ndarray:
        """Generate spatially correlated noise for 1D"""
        # Generate white noise
        noise = jr.normal(key, shape)
        
        # Apply spatial smoothing with 1D kernel
        kernel = jnp.array([0.0, 1.0, 0.0])  # 1D smoothing kernel
        
        # Pad for convolution with periodic boundary
        noise_padded = jnp.pad(noise, 1, mode='wrap')
        
        # Apply 1D convolution
        smoothed_noise = jnp.zeros_like(noise)
        for i in range(3):
            smoothed_noise += kernel[i] * noise_padded[i:i+self.nx]
        
        return smoothed_noise
    
    def drift(self, u, D):
        """True reaction term to be discovered
        
        Examples:
        - Fisher-KPP: r*u*(1-u/K)
        """
        # Fisher-KPP type reaction: u*(1-u)
        if self.true_system:
            laplacian_u = self.laplacian(u)
            diffusion = D * laplacian_u
            reaction = u * (1 - u)
        else:
            laplacian_u = self.laplacian(u)
            diffusion = D * laplacian_u
            reaction = u*(0.929 - 0.936*u)
        return diffusion + reaction

    def noise_diffusion(self, x, t, u):
        """True noise coefficient function to be discovered"""
        if self.true_system:
            return 0.1 * jnp.abs(u)  # Multiplicative noise
        else:
            return 0.0935*jnp.abs(u) - 0.0117

    def generate_spde_data(
        self,
        key,
        D: float = 0.1,
        T: float = 10.0,
        n_trajectories: int = 3,
        save_every: int = 20,
    ) -> Dict[str, jnp.ndarray]:
        """Generate synthetic 1D reaction-diffusion data (vectorized over trajectories)."""

        n_steps = int(T / self.dt)
        save_idx = jnp.arange(0, n_steps, save_every)
        n_save = save_idx.shape[0]

        traj_keys = jr.split(key, n_trajectories)

        def simulate_one(traj_key):
            u0 = jnp.sin(self.x) + 1.0

            def step_fn(carry, step):
                key_i, u = carry
                key_i, noise_key = jr.split(key_i)

                t = step * self.dt
                deterministic = self.drift(u, D)
                noise = self.generate_noise(noise_key, (self.nx,))
                noise_coeff = self.noise_diffusion(self.x, t, u)
                stochastic = noise_coeff * noise * jnp.sqrt(self.dt)

                u_next = u + self.dt * deterministic + stochastic
                return (key_i, u_next), u_next

            (_, _), u_hist = jax.lax.scan(step_fn, (traj_key, u0), jnp.arange(n_steps))
            return u_hist[save_idx]  # (n_save, nx)

        all_u = jax.vmap(simulate_one)(traj_keys)  # (n_trajectories, n_save, nx)
        all_times = save_idx * self.dt

        return all_u, self.x, all_times