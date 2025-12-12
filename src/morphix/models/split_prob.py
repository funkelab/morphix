import equinox as eqx
import jax

from ..cell import Cell


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
        self.eps = float(eps)
        self.layers = (
            eqx.nn.Linear(cell_state_dims, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dims, 1, key=key2),
            jax.nn.sigmoid,
        )

    def __call__(self, cells: Cell, key, extended_attributes: bool = False):
        # compute split probabilities
        p_split = jax.vmap(self.split_probs)(cells.state)

        # sample an action (split or not)
        split = self.sample(p_split, key)

        return cells.replace(p_split=p_split, split=split)

    def split_probs(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        return x.squeeze(-1)

    def sample(self, split_probs, key):
        key1, key2 = jax.random.split(key, 2)

        uniform = jax.random.uniform(key1, shape=split_probs.shape)

        if self.eps > 0:
            explore = jax.random.uniform(key2) < self.eps
            split_probs = 0.5 * explore + split_probs * (1 - explore)

        return split_probs > uniform
