import jax
import jax.random as jr
import jax.numpy as jnp
import diffrax

def simulate(env, init_state, ts, process_noise_key):

    def drift(t, x, args):
        dx = env.drift(t, x, args)
        return dx
    
    def diffusion(t, x, args):
        return env.diffusion(t, x, args)

    brownian_motion = diffrax.UnsafeBrownianPath(shape=(env.n_var,), key=process_noise_key, levy_area=diffrax.SpaceTimeLevyArea) #define process noise
    system = diffrax.MultiTerm(diffrax.ODETerm(drift), diffrax.ControlTerm(diffusion, brownian_motion))

    sol = diffrax.diffeqsolve(
        system,
        diffrax.GeneralShARK(),
        t0=ts[0],
        t1=ts[-1],
        dt0=0.001,
        y0=init_state,
        saveat=diffrax.SaveAt(ts=ts),
        adjoint=diffrax.DirectAdjoint(),
        max_steps = 16**5
    )

    ys = sol.ys
    return ys

def generate_data(key, env, dt, T, batch_size):
    key, init_key, p1_key, ts_key = jr.split(key, 4)

    ts = jnp.tile(jnp.arange(0, T, dt), (batch_size, 1))

    init_states = env.sample_init_states(batch_size, init_key)
    process_noise_keys = jr.split(p1_key, batch_size)

    ys = jax.vmap(simulate, in_axes=(None, 0, 0, 0))(env, init_states, ts, process_noise_keys)

    return ts, ys