"""Test basic initialization."""

from quantum_tunneling.scattering_sim import ScatteringSim


def test_init():
    x_min = -1.0
    x_max = 1.0
    x_res = 100
    scattering_sim = ScatteringSim(x_min=-1.0, x_max=1.0, x_res=100)
    assert scattering_sim.x_max == x_max
    assert scattering_sim.x_min == x_min
    assert scattering_sim.x_res == x_res
    assert scattering_sim.x_range.min() == x_min
    assert scattering_sim.x_range.max() == x_max
    assert scattering_sim.x_range.shape[0] == x_res
