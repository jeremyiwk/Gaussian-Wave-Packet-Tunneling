"""Check expected behavior of derivative functions."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from quantum_tunneling.derivatives import finite_difference_d2


@pytest.mark.parametrize(
    "func",
    [
        jnp.cos,
        jnp.sin,
        jnp.square,
        jnp.exp,
        jnp.tanh,
        jnp.sinh,
        jnp.cosh,
        jnp.arctan,
        jnp.arctanh,
    ],
)
class TestDerivatives:
    """Class to hold derivative test data."""

    x_min: float = -1.0
    x_max: float = 1.0
    x_res: int = 256
    x_range = jnp.linspace(x_min, x_max, x_res)
    dx = x_range[1] - x_range[0]

    def test_finite_difference(self, func: Callable) -> None:
        """Check accuracy of finite differences against autodiff."""
        f = func(self.x_range)
