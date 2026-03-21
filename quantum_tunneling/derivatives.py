"""Various implementations of derivative operators."""

import jax
import jax.numpy as jnp


def finite_difference_d2(f: jax.Array, dx: float) -> jax.Array:
    """Finite difference 2nd derivative via jnp.gradient `f`."""
    return jnp.gradient(jnp.gradient(f, axis=(-1,)), axis=(-1,)) / dx**2
