import equinox as eqx
import jax

from ..cell import Cell
from ..diffusion import steady_state_concentrations


class DiffusionModel(eqx.Module):
    diffusion_coefs: jax.Array
    degradation_rates: jax.Array

    def __init__(self, diffusion_coefs, degradation_rates):
        self.diffusion_coefs = diffusion_coefs
        self.degradation_rates = degradation_rates

    def __call__(self, cells: Cell):
        concentrations = steady_state_concentrations(
            cells.position,
            cells.radius,
            cells.secretion,
            self.diffusion_coefs,
            self.degradation_rates,
            cells.active,
        )
        return cells.replace(concentration=concentrations)
