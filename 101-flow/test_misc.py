import jax
import jax.numpy as jnp
import numpy as np

from utils_nn import *


def test_integrate_midpoint_rule_scalar():
    # f(t) = t^2, integral from 0 to 1 is 1/3
    f = lambda t: t**2

    t_grid = jnp.linspace(0.0, 1.0, 1001)
    result = integrate(f, t_grid)

    expected = 1.0 / 3.0
    assert jnp.allclose(result, expected, atol=1e-4)


def test_integrate_midpoint_rule_vector_output():
    # f(t) returns a vector: [t, t^2]
    f = lambda t: jnp.array([t[0], t[0] ** 2])

    t_grid = jnp.linspace(0.0, 2.0, 2001)
    result = integrate(f, t_grid)

    # Analytical integrals:
    # ∫ t dt from 0 to 2 = 2
    # ∫ t^2 dt from 0 to 2 = 8/3
    expected = jnp.array([2.0, 8.0 / 3.0])

    assert result.shape == (2,)
    assert jnp.allclose(result, expected, atol=1e-4)


def test_integrate_vector_complex_function():
    # f(t) = [sin(t), exp(-t)], integral from 0 to 1
    f = lambda t: jnp.array([jnp.sin(t[0]), jnp.exp(-t[0])])

    t_grid = jnp.linspace(0.0, 1.0, 1001)
    result = integrate(f, t_grid)

    # Analytical integrals:
    # ∫ sin(t) dt from 0 to 1 = 1 - cos(1)
    # ∫ exp(-t) dt from 0 to 1 = 1 - 1/e
    expected = jnp.array([1 - jnp.cos(1.0), 1 - 1 / jnp.e])

    assert result.shape == (2,)
    assert jnp.allclose(result, expected, atol=1e-4)


def test_integrate_constant_function():
    # f(t) = c, integral should be c * (b - a)
    c = jnp.array([3.0, -1.0])
    f = lambda t: c

    t_grid = jnp.linspace(-1.0, 4.0, 1001)
    result = integrate(f, t_grid)

    expected = c * (4.0 - (-1.0))
    assert jnp.allclose(result, expected, atol=1e-15)


if __name__ == "__main__":
    test_integrate_midpoint_rule_scalar()
    test_integrate_midpoint_rule_vector_output()
    test_integrate_vector_complex_function()
    test_integrate_constant_function()

    print("\nDone!")
