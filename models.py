from flax import nnx
import jax


class ReactModel(nnx.Module):
    def __init__(self, cell_state_dims: int, hidden_dims: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(cell_state_dims, hidden_dims, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(hidden_dims, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dims, cell_state_dims, rngs=rngs)

    def __call__(self, cell_state: jax.Array):
        x = self.linear1(cell_state)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = nnx.sigmoid(x)
        return x


class SplitModel(nnx.Module):
    def __init__(self, cell_state_dims: int, hidden_dims: int, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.linear1 = nnx.Linear(cell_state_dims, hidden_dims, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(hidden_dims, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dims, 1, rngs=rngs)

    def __call__(self, cell_state: jax.Array):
        x = self.linear1(cell_state)
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = nnx.sigmoid(x)
        return x.squeeze(-1)

    def sample(self, split_probs):
        uniform = jax.random.uniform(self.rngs.split_probs(), shape=split_probs.shape)
        return split_probs > uniform
