"""Test basic initialization."""

import jax.numpy as jnp

from quantum_tunneling.scattering_sim import ScatteringSim


def test_init() -> None:
    """Test basic initialization of sim object."""
    x_min, x_max, x_res = x_domain = -1.0, 1.0, 100
    t_domain = 0.0, 1.0, 0.01

    def ic(x: float) -> float:
        x**2

    def potential(x: float) -> float:
        jnp.clip(x, -0.1, 0.1)

    scattering_sim = ScatteringSim(
        x_domain=x_domain,
        t_domain=t_domain,
        initial_condition=ic,
        potential=potential,
    )

    assert scattering_sim.x_max == x_max
    assert scattering_sim.x_min == x_min
    assert scattering_sim.x_res == x_res
    assert scattering_sim.x_range.min() == x_min
    assert scattering_sim.x_range.max() == x_max
    assert scattering_sim.x_range.shape[0] == x_res
