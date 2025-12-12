import equinox as eqx
import jax

from ..cell import Cell


class SecretionModel(eqx.Module):
    num_molecules: int
    layers: tuple

    def __init__(self, num_molecules: int, cell_state_dims: int, hidden_dims: int, key):
        self.num_molecules = num_molecules

        key1, key2 = jax.random.split(key, 2)
        self.layers = (
            eqx.nn.Linear(cell_state_dims, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dims, num_molecules, key=key2),
            # ensure that secretion values are positive
            jax.nn.relu,
        )

    def __call__(self, cells: Cell):
        secretion = jax.vmap(self.compute_secretion)(cells.state)
        return cells.replace(secretion=secretion)

    def compute_secretion(self, state):
        x = state
        for layer in self.layers:
            x = layer(x)
        return x
