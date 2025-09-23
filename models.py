import equinox as eqx
from cell import Cell
import jax
import jax.numpy as jnp


class ReactModel(eqx.Module):
    layers: tuple

    def __init__(self, cell_state_dims: int, hidden_dims: int, key):
        key1, key2 = jax.random.split(key, 2)
        self.layers = (
            eqx.nn.Linear(cell_state_dims, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dims, cell_state_dims, key=key2),
            jax.nn.sigmoid,
        )

    def __call__(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        return cell_state + x


class SplitModel(eqx.Module):
    eps: float
    layers: tuple

    def __init__(
        self,
        cell_state_dims: int,
        hidden_dims: int,
        key,
        eps: float = 0.0,
    ):
        """Compute the probability of a split event from the cell state.

        Args:

            cell_state_dims:

                The size of the cell state.

            hidden_dims:

                The size of the hidden layer of the MLP.

            key:

                RNG key to use for initialization.

            eps:

                The probability to act randomly (split prob will be 0.5), to be
                used to incentivise exploration during learning.
        """
        key1, key2 = jax.random.split(key, 2)
        self.eps = eps
        self.layers = (
            eqx.nn.Linear(cell_state_dims, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dims, 1, key=key2),
            jax.nn.sigmoid,
        )

    def __call__(self, cell_state: jax.Array, key):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        if self.eps > 0:
            explore = jax.random.uniform(key) < self.eps
            x = 0.5 * explore + x * (1 - explore)
        return x.squeeze(-1)

    def sample(self, split_probs, key):
        uniform = jax.random.uniform(key, shape=split_probs.shape)
        return split_probs > uniform


class Model(eqx.Module):
    max_num_cells: int
    cell_state_dims: int
    initial_cell_states: jax.Array
    react_model: eqx.Module
    split_model: eqx.Module

    def __init__(self, max_num_cells, cell_state_dims, react_model, split_model, key):
        self.max_num_cells = max_num_cells
        self.cell_state_dims = cell_state_dims
        self.initial_cell_states = jax.random.uniform(
            key=key, shape=(max_num_cells, cell_state_dims)
        )
        self.react_model = react_model
        self.split_model = split_model

    def initialize_cells(self, key):
        key1, key2 = jax.random.split(key, 2)
        keys = jax.random.split(key1, self.max_num_cells)

        cell_states = self.initial_cell_states
        p_split = jax.vmap(self.split_model)(cell_states, keys)
        split = self.split_model.sample(p_split, key=key2)

        return Cell(
            position=jnp.zeros((self.max_num_cells, 3)),
            state=cell_states,
            # initially, only one cell is active
            parent=(-jnp.ones((self.max_num_cells,), dtype=jnp.int16)).at[0].set(0),
            p_split=p_split,
            split=split,
        )

    def partition(self):
        """Partition the model into (parameters, static)."""
        return eqx.partition(self, eqx.is_inexact_array)
