"""Random utility functions."""

import jax
import jax.numpy as jnp


def l2_norm(fx: jax.Array, dx: float) -> float:
    """Dot product with dx scale."""
    return jnp.linalg.norm(fx) * dx


def l2_rel_err(fx: jax.Array, gx: jax.Array) -> float:
    """L2 relative error between two arrays."""
    return l2_norm(fx - gx, 1.0) / l2_norm(fx, 1.0)
