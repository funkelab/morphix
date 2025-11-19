from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import optax

from .cell import Cell
from .reinforce import reinforcement_losses
from .simulation import simulate


class BatchLog(eqx.Module):
    """A data class to store training details for a single batch."""

    _raw_losses: jax.Array
    _rl_losses: jax.Array
    losses: jax.Array
    trajectory: Cell

    @property
    def raw_losses(self):
        """The raw losses (per time step and cell)."""
        return self._raw_losses * self.trajectory.active

    @property
    def raw_loss(self):
        """The overall raw loss (averaged over active cells)."""
        return self.raw_losses.sum() / self.trajectory.active.sum()

    @property
    def rl_losses(self):
        """The reinforcement learning losses (per time step and cell)."""
        return self._rl_losses * self.trajectory.active

    @property
    def rl_loss(self):
        """The overall reinforcement learning loss (averaged over active cells)."""
        return self.rl_losses.sum() / self.trajectory.active.sum()

    @property
    def loss(self):
        """The combined loss (raw and RL, averaged over active cells)."""
        return self.losses.sum() / self.trajectory.active.sum()


@partial(
    jax.jit,
    static_argnames=(
        "static",
        "loss_function",
        "num_timesteps",
        "optimizer",
        "batch_size",
        "debug",
    ),
)
def train_step(
    params: optax.Params,
    static: eqx.Module,
    loss_function: Callable[[Cell, float], tuple[jax.Array, jax.Array]],
    num_timesteps: int,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    key: jax.Array,
    batch_size: int = 1,
    debug: bool = False,
):
    """Perform a single training step.

    Args:
        params:

            The trainable parameters of the model.

        static:

            All other static variables of the model.

        loss_function:

            A custom loss function to compute the lineage and properties losses
            for a given timestep.

            The loss function takes `(cells, t)` as input. `cells` is the set
            of all cells at time `t`. `cells` includes inactive cells. `t` is
            the simulation time (`timestep * model.delta_t`).

            It has to return a loss array that should have shape `(n,)`, where
            `n` is the number of cells (including inactive cells). Losses for
            inactive cells can be arbitrary and will be ignored (zeroed-out
            later on).

            This loss should serve two purposes: it should score the topology
            of the lineage and attributes of the cells (like position and
            size). The loss will be directly optimized using gradient descent
            (to fit attributes) and also through reinforcement learning on cell
            split decisions (to obtain a correct lineage tree).

        num_timesteps:

            The number of timesteps to simulate.

        optimizer:

            The optax optimizer and state to use.

        opt_state:

            The optax optimizer and state to use.

        key:

            JAX random key.

        batch_size:

            The number of simulation runs to average before updating the
            gradient. Increasing the batch size can help decrease the variance
            of the reinforcement learning loss.

        debug:

            If set, run in debug mode and print extra information.

    """

    @partial(jax.jit, static_argnames=("static",))
    @partial(jax.value_and_grad, has_aux=True)
    def batch_loss_grad(params, static, key):
        keys = jax.random.split(key, batch_size)
        model = eqx.combine(params, static)
        loss, batch_log = jax.vmap(
            lambda k: simulation_loss(model, loss_function, num_timesteps, k, debug)
        )(keys)
        return loss.mean(), batch_log

    (loss, batch_log), grad = batch_loss_grad(params, static, key)

    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    return loss, params, opt_state, batch_log


def simulation_loss(
    model,
    loss_function,
    num_timesteps: int,
    key,
    debug: bool = False,
) -> tuple[jax.Array, BatchLog]:
    """Run the simulation and compute the loss for the given loss function."""
    # run the simulation
    trajectory = simulate(model, num_timesteps, key, debug)

    # compute the loss
    loss, batch_log = trajectory_loss(
        trajectory,
        loss_function,
        model.rl_discount_gamma,
        model.entropy_regularizer,
        model.direct_loss_weight,
    )

    return loss, batch_log


def trajectory_loss(
    trajectory: Cell,
    loss_function,
    rl_discount_gamma,
    entropy_regularizer,
    direct_loss_weight,
) -> tuple[jax.Array, BatchLog]:
    """Compute the loss for an entire trajectory."""
    # compute user-provided losses
    raw_losses = loss_function(trajectory)

    # turn losses into reinforcement learning losses
    rl_losses = reinforcement_losses(
        trajectory,
        raw_losses,
        gamma=rl_discount_gamma,
        entropy_regularizer=entropy_regularizer,
    )

    # and add those to the raw losses
    losses = direct_loss_weight * raw_losses + rl_losses

    # zero-out losses of inactive cells
    active = trajectory.active
    losses *= active

    # compute mean loss over all active cells
    loss = losses.sum() / active.sum()

    batch_log = BatchLog(
        _raw_losses=raw_losses,
        _rl_losses=rl_losses,
        losses=losses,
        trajectory=trajectory,
    )

    return loss, batch_log
