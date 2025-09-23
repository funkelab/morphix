from flax import nnx
from cell import Cell
import jax
import jax.numpy as jnp


class ReactModel(nnx.Module):
    def __init__(self, cell_state_dims: int, hidden_dims: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(cell_state_dims, hidden_dims, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(hidden_dims, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dims, cell_state_dims, rngs=rngs)

    def __call__(self, cell_state: jax.Array):
        x = self.linear1(cell_state)
        x = self.layer_norm(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.sigmoid(x)
        return cell_state + x


class SplitModel(nnx.Module):
    def __init__(
        self,
        cell_state_dims: int,
        hidden_dims: int,
        eps: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Compute the probability of a split event from the cell state.

        Args:

            cell_state_dims:

                The size of the cell state.

            hidden_dims:

                The size of the hidden layer of the MLP.

            eps:

                The probability to act randomly (split prob will be 0.5), to be
                used to incentivise exploration during learning.

            rngs:

                Random number generator with streams for "params" and
                "split_probs".
        """
        self.rngs = rngs
        self.eps = eps
        self.linear1 = nnx.Linear(cell_state_dims, hidden_dims, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(hidden_dims, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dims, 1, rngs=rngs)

    def __call__(self, cell_state: jax.Array):
        x = self.linear1(cell_state)
        x = self.layer_norm(x)
        x = nnx.relu(x)
        x = self.linear2(x)
        x = nnx.sigmoid(x)
        if self.eps > 0:
            explore = self.rngs.split_probs.uniform(x.shape) < self.eps
            x = 0.5 * explore + x * (1 - explore)
        return x.squeeze(-1)

    def sample(self, split_probs):
        uniform = jax.random.uniform(self.rngs.split_probs(), shape=split_probs.shape)
        return split_probs > uniform


class Model(nnx.Module):
    def __init__(
        self, max_num_cells, cell_state_dims, react_model, split_model, *, rngs
    ):
        self.max_num_cells = max_num_cells
        self.cell_state_dims = cell_state_dims
        self.initial_cell_state = nnx.Param(
            rngs.uniform((max_num_cells, cell_state_dims))
        )
        self.react_model = react_model
        self.split_model = split_model

    def initialize_cells(self):
        initial_cells = Cell(
            position=jnp.zeros((self.max_num_cells, 3)),
            state=self.initial_cell_state.value,
            # initially, only one cell is active
            parent=(-jnp.ones((self.max_num_cells,), dtype=jnp.int16)).at[0].set(0),
            p_split=self.split_model(self.initial_cell_state),
            split=None,
        )
        initial_cells = initial_cells.replace(
            split=self.split_model.sample(initial_cells.p_split)
        )
        return initial_cells

    def split(self):
        return nnx.split(self, nnx.Param, ...)

    @classmethod
    def create(cls, definition, params, state):
        return nnx.merge(definition, params, state)

    def update_params(self, params):
        model_def, _, state = nnx.split(self, nnx.Param, ...)
        return nnx.merge(model_def, params, state)
