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

    lineage_loss: jax.Array
    _cell_losses: jax.Array
    _rl_losses: jax.Array
    losses: jax.Array
    trajectory: Cell

    @property
    def cell_losses(self):
        return self._cell_losses * self.trajectory.active

    @property
    def rl_losses(self):
        return self._rl_losses * self.trajectory.active

    @property
    def cell_loss(self):
        return self.cell_losses.sum() / self.trajectory.active.sum()

    @property
    def rl_loss(self):
        return self.rl_losses.sum() / self.trajectory.active.sum()


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

            It has to return two loss values: `(lineage_loss, cell_losses)`.
            `lineage_loss` is a scalar and `cell_losses` should have shape
            `(n,)`, where `n` is the number of cells (including inactive
            cells). Losses for inactive cells can be arbitrary and will be
            ignored (zeroed-out later on).

            The lineage loss scores the topology of the lineage and will be
            used in a reinforcement learning method to adjust the split
            probabilities of the model. This loss does not have to be
            differentiable.

            The cell losses are differentiable losses on arbitrary attributes
            of the cells (e.g., their position, size, or state). Those losses
            will be directly optimized.

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
    )

    return loss, batch_log


def trajectory_loss(
    trajectory: Cell,
    loss_function,
) -> tuple[jax.Array, BatchLog]:
    """Compute the loss for an entire trajectory."""
    # compute user-provided lineage and cell losses
    lineage_losses, cell_losses = loss_function(trajectory)

    # turn (non-differentiable) lineage losses into reinforcement learning
    # losses
    rl_losses = reinforcement_losses(trajectory, lineage_losses)

    # combine losses
    losses = rl_losses + cell_losses

    # zero-out losses of inactive cells
    active = trajectory.active
    losses *= active

    # compute mean loss over all active cells
    loss = losses.sum() / active.sum()

    batch_log = BatchLog(
        lineage_loss=lineage_losses,
        _cell_losses=cell_losses,
        _rl_losses=rl_losses,
        losses=losses,
        trajectory=trajectory,
    )

    return loss, batch_log
