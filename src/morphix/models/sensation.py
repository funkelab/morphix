import equinox as eqx
import jax
import jax.numpy as jnp

from ..cell import Cell


class SensationModel(eqx.Module):
    layers: tuple

    def __init__(self, cell_state_dims, num_molecules, hidden_dims, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = (
            eqx.nn.Linear(cell_state_dims + num_molecules, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dims, cell_state_dims, key=key2),
        )

    def __call__(self, cells: Cell):
        state = jax.vmap(self.update_state)(cells.state, cells.concentration)
        return cells.replace(state=state)

    def update_state(self, state, concentration):
        x = jnp.concatenate((state, concentration))
        for layer in self.layers:
            x = layer(x)
        return state + x
