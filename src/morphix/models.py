import equinox as eqx
import jax
import jax.numpy as jnp

from .cell import Cell


def create_model(max_num_cells, cell_state_dims, exploration_eps, key):
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)

    react_model = ReactModel(
        cell_state_dims,
        cell_state_dims * 2,
        key=key1,
    )
    move_model = MoveModel(
        cell_state_dims,
        cell_state_dims * 2,
        3,
        key=key2,
    )
    split_prob_model = SplitProbModel(
        cell_state_dims,
        cell_state_dims * 2,
        eps=exploration_eps,
        key=key3,
    )

    split_model = SplitModel(
        cell_state_dims,
        3,
        cell_state_dims * 2,
        key=key4,
    )

    model = Model(
        max_num_cells,
        cell_state_dims,
        react_model,
        move_model,
        split_prob_model,
        split_model,
        key=key5,
    )
    return model


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


class MoveModel(eqx.Module):
    layers: tuple
    spatial_dims: int

    def __init__(self, cell_state_dims: int, hidden_dims: int, spatial_dims: int, key):
        key1, key2 = jax.random.split(key, 2)
        self.spatial_dims = spatial_dims
        self.layers = (
            eqx.nn.Linear(cell_state_dims, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            # predict mean and log of standard deviation
            eqx.nn.Linear(hidden_dims, spatial_dims * 2, key=key2),
        )

    def __call__(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        mean = x[: self.spatial_dims]
        # clip the standard deviation to avoid distribution collapse
        std = jnp.clip(jnp.exp(x[self.spatial_dims :]), min=1e-6)
        return mean, std

    def sample(self, mean, std, key, return_log_p=False):
        move = mean + std * jax.random.normal(key, shape=mean.shape)
        if not return_log_p:
            return move
        var = std**2
        log_p_move = (
            -0.5 * (jnp.log(2 * jnp.pi * var) + ((move - mean) ** 2) / var)
        ).sum(axis=-1)
        return move, log_p_move


class SplitProbModel(eqx.Module):
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


class SplitModel(eqx.Module):
    layers: tuple
    cell_state_dims: int
    spatial_dims: int

    def __init__(
        self,
        cell_state_dims: int,
        spatial_dims: int,
        hidden_dims: int,
        key,
    ):
        """Predict how to divide a cell.

        Given a cell state, output coefficients for how to split the state into
        two daughter cells, the ratio of their sizes, and the orientation of
        the division plane.

        Args:
            cell_state_dims:

                The size of the cell state.

            spatial_dims:

                The number of spatial dimensions (used to predict the division
                plane).

            hidden_dims:

                The size of the hidden layer of the MLP.

            key:

                RNG key to use for initialization.
        """
        key1, key2 = jax.random.split(key, 2)
        self.layers = (
            eqx.nn.Linear(cell_state_dims, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dims, cell_state_dims + 1 + spatial_dims, key=key2),
        )
        self.cell_state_dims = cell_state_dims
        self.spatial_dims = spatial_dims

    def __call__(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        state_ratio = jax.nn.sigmoid(x[: self.cell_state_dims])
        volume_ratio = jax.nn.sigmoid(x[self.cell_state_dims])
        division_plane = x[self.cell_state_dims + 1 :]
        division_plane = division_plane / jnp.clip(
            jnp.linalg.norm(division_plane), min=1e-10
        )
        return state_ratio, volume_ratio, division_plane


class Model(eqx.Module):
    max_num_cells: int
    cell_state_dims: int
    initial_cell_states: jax.Array
    react_model: eqx.Module
    move_model: eqx.Module
    split_prob_model: eqx.Module
    split_model: eqx.Module

    def __init__(
        self,
        max_num_cells,
        cell_state_dims,
        react_model,
        move_model,
        split_prob_model,
        split_model,
        key,
    ):
        self.max_num_cells = max_num_cells
        self.cell_state_dims = cell_state_dims
        self.initial_cell_states = jax.random.uniform(
            key=key, shape=(max_num_cells, cell_state_dims)
        )
        self.react_model = react_model
        self.move_model = move_model
        self.split_prob_model = split_prob_model
        self.split_model = split_model

    def initialize_cells(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        keys = jax.random.split(key1, self.max_num_cells)

        cell_states = self.initial_cell_states

        # initial positions
        mean, std = jax.vmap(self.move_model)(cell_states)
        keys = jax.random.split(key2, self.max_num_cells)
        position, log_p_move = jax.vmap(
            lambda m, v, k: self.move_model.sample(m, v, k, return_log_p=True)
        )(mean, std, keys)

        # initial split decisions
        p_split = jax.vmap(self.split_prob_model)(cell_states, keys)
        split = self.split_prob_model.sample(p_split, key=key3)

        return Cell(
            log_p_move=log_p_move,
            position=position,
            radius=jnp.ones((self.max_num_cells,), dtype=jnp.float32),
            state=cell_states,
            # initially, only one cell is active
            parent=(-jnp.ones((self.max_num_cells,), dtype=jnp.int16)).at[0].set(0),
            p_split=p_split,
            split=split,
        )

    def partition(self):
        """Partition the model into (parameters, static)."""
        return eqx.partition(self, eqx.is_inexact_array)
