"""Defines a scattering simulation object."""

from __future__ import annotations

from typing import Function

import equinox as eqx
import jax
import jax.numpy as jnp


class ScatteringSim(eqx.Module):
    """Creates a simulation object."""

    x_min: float
    x_max: float
    x_res: int
    x_range: jax.Array
    t_min: float
    t_max: float
    dt_max: float
    potential: jax.Array
    initial_condition: jax.Array

    def __init__(
        self: ScatteringSim,
        potential: Function,
        initial_condition: Function,
        x_domain: tuple[float, float, int],
        t_domain: tuple[float, float, float],
    ) -> None:
        """Initialize simulation object."""
        self.x_min, self.x_max, self.x_res = x_domain
        self.x_range = jnp.linspace(*x_domain)
        self.t_min, self.t_max, self.dt_max = t_domain
        self.potential = potential(self.x_range)
        self.initial_condition = initial_condition(self.x_range)
