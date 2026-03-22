from quantum_tunneling.derivatives import finite_difference_d2, spectral_d2

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

x_min = -jnp.pi * 10.0
x_max = jnp.pi * 10.0
x_res = 256

x_range = jnp.linspace(x_min, x_max, x_res)
dx = x_range[1] - x_range[0]

f = lambda x: jnp.sin(x)
d2f = lambda x: -jnp.sin(x)

fx = f(x_range)
d2fx = d2f(x_range)

num_d2fx = finite_difference_d2(fx, dx)

# num_d2fx = spectral_d2(fx, dx)

err = jnp.abs(num_d2fx - d2fx)

# plt.plot(x_range, fx)


# plt.plot(x_range, d2fx, label="sym")
plt.plot(x_range, num_d2fx, label="num")

# plt.plot(x_range, err)
# plt.yscale('log')
plt.legend()
plt.show()
