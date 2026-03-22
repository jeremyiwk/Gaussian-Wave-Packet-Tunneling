"""Defines a scattering simulation object."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

from quantum_tunneling.derivatives import spectral_d2

_DEFAULT_SOLVER = dfx.Dopri5
_DEFAULT_STEPSIZE_CONTROLLER = dfx.PIDController(rtol=1e-5, atol=1e-5)


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
    term: dfx.ODETerm
    solver: dfx.AbstractSolver
    stepsize: dfx.AbstractStepSizeController

    def __init__(
        self: ScatteringSim,
        potential: Callable,
        initial_condition: Callable,
        x_domain: tuple[float, float, int],
        t_domain: tuple[float, float, float],
        solver: dfx.AbstractSolver = _DEFAULT_SOLVER,
        stepsize: dfx.AbstractStepSizeController = _DEFAULT_STEPSIZE_CONTROLLER,
    ) -> None:
        """Initialize simulation object."""
        self.x_min, self.x_max, self.x_res = x_domain
        self.x_range = jnp.linspace(*x_domain)
        self.t_min, self.t_max, self.dt_max = t_domain
        self.potential = potential(self.x_range)
        self.initial_condition = initial_condition(self.x_range)

        self.term = dfx.ODETerm(self.schrodinger)
        self.solver = solver
        self.stepsize = stepsize

    @jax.jit
    def schrodinger(self, t: float, psi: jax.Array, args: tuple[Any, ...]) -> jax.Array:  # noqa: ARG002
        """RHS of schrodinger eq."""
        return 1j * (spectral_d2(psi, self.dx) + self.potential * psi)

    def solve(self) -> jax.Array:
        """Solve the simulation."""
        t0, t1 = self.t_max, self.t_max
        dt = self.dt_max
        return dfx.diffeqsolve(
            self.term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=dt,
            y0=self.initial_condition,
            stepsize_controller=self.stepsize,
        )
