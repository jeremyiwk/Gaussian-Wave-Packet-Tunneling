"""Various implementations of derivative operators."""

import jax
import jax.numpy as jnp

_FINITE_DIFF_D2 = jnp.array([1.0, -2.0, 1.0])



def finite_difference_d2(f: jax.Array, dx: float) -> jax.Array:
    """Finite difference 2nd derivative via convolution `f`."""
    kernel = jnp.tile(_FINITE_DIFF_D2, reps=(*f.shape[:-1], 1))
    return jax.scipy.signal.convolve(f, kernel, mode="same") / dx**2

def spectral_d2(f: jax.Array, dx: float) -> jax.Array:
    """Spectral derivative of `f` via fft."""
    k = 2 * jnp.pi * jnp.fft.rfftfreq(f.shape[-1], dx)
    return - jnp.fft.irfft(k ** 2 * jnp.fft.rfft(f))
