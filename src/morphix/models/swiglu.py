import equinox as eqx
import jax
import jax.numpy as jnp


class SwiGLU(eqx.Module):
    layer_1: eqx.nn.Linear
    layer_2: eqx.nn.Linear

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        out_features: int | None = None,
        use_bias: bool = True,
        key: jax.Array,
    ):
        if out_features is None:
            out_features = in_features

        key1, key2 = jax.random.split(key)

        # input projection
        self.layer_1 = eqx.nn.Linear(
            in_features,
            2 * hidden_features,
            use_bias=use_bias,
            key=key1,
        )

        # output projection
        self.layer_2 = eqx.nn.Linear(
            hidden_features,
            out_features,
            use_bias=use_bias,
            key=key2,
        )

    def __call__(self, x: jax.Array) -> jnp.ndarray:
        proj = self.layer_1(x)
        a, b = jnp.split(proj, 2, axis=-1)
        gated = a * jax.nn.swish(b)
        return self.layer_2(gated)
