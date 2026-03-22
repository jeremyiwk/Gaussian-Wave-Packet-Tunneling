"""Check expected behavior of derivative functions."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from quantum_tunneling.derivatives import spectral_d2
from quantum_tunneling.utils import l2_rel_err

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


def _gaussian(x: float) -> float:
    return jnp.exp(-(x**2))


def _d2_gaussian(x: float) -> float:
    return (4 * x**2 - 2) * _gaussian(x)


@pytest.mark.parametrize(
    "func",
    [
        (_gaussian, _d2_gaussian),
    ],
)
class TestDerivatives:
    """Class to hold derivative test data."""

    x_min: float = -jnp.pi * 5.0
    x_max: float = jnp.pi * 5.0
    x_res: int = 512
    x_range = jnp.linspace(x_min, x_max, x_res)
    dx = x_range[1] - x_range[0]

    def test_spectral(self, func: tuple[Callable, Callable]) -> None:
        """Check accuracy of spectral deriv."""
        f, df = func
        fx = f(self.x_range)
        sym_df = df(self.x_range)
        num_df = spectral_d2(fx, self.dx)
        err = l2_rel_err(sym_df, num_df)

        assert jnp.isclose(err, 0.0, atol=1e-4)
