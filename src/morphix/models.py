import equinox as eqx
import jax
import jax.numpy as jnp

from .cell import Cell
from .diffusion import steady_state_concentrations
from .mechanics import morse_force


def create_model(
    key,
    max_num_cells,
    cell_state_dims,
    num_molecules,
    delta_t,
    exploration_eps=0.0,
    morse_well_width=1.0,
    morse_well_depth=1.0,
):
    key1, key2, key3, key4, key5, key6, key7 = jax.random.split(key, 7)

    react_model = ReactModel(
        cell_state_dims,
        cell_state_dims * 2,
        key=key1,
    )
    motility_model = MotilityModel(
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

    mechanics_model = MechanicsModel(morse_well_width, morse_well_depth)

    secretion_model = SecretionModel(
        num_molecules, cell_state_dims, cell_state_dims * 2, key5
    )
    diffusion_model = DiffusionModel(
        diffusion_coefs=jnp.ones(num_molecules),
        degradation_rates=jnp.ones(num_molecules),
    )
    sensation_model = SensationModel(
        cell_state_dims, num_molecules, hidden_dims=cell_state_dims * 2, key=key6
    )

    model = Model(
        max_num_cells=max_num_cells,
        cell_state_dims=cell_state_dims,
        num_molecules=num_molecules,
        delta_t=delta_t,
        react_model=react_model,
        motility_model=motility_model,
        split_prob_model=split_prob_model,
        split_model=split_model,
        mechanics_model=mechanics_model,
        secretion_model=secretion_model,
        diffusion_model=diffusion_model,
        sensation_model=sensation_model,
        key=key7,
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

    def __call__(self, cells: Cell, extended_attributes: bool = False):
        state = jax.vmap(self.update_state)(cells.state)
        return cells.replace(state=state)

    def update_state(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        return cell_state + x


class MotilityModel(eqx.Module):
    layers: tuple
    spatial_dims: int

    def __init__(self, cell_state_dims: int, hidden_dims: int, spatial_dims: int, key):
        key1, key2 = jax.random.split(key, 2)
        self.spatial_dims = spatial_dims
        self.layers = (
            eqx.nn.Linear(cell_state_dims, hidden_dims, key=key1),
            eqx.nn.LayerNorm(hidden_dims),
            jax.nn.relu,
            eqx.nn.Linear(hidden_dims, spatial_dims, key=key2),
        )

    def __call__(self, cells, extended_attributes: bool = False):
        force = jax.vmap(self.motility_force)(cells.state)
        return cells.replace(motility_force=force)

    def motility_force(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        return x


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


class SplitModel(eqx.Module):
    layers: tuple
    cell_state_dims: int
    spatial_dims: int
    max_volume_difference: float

    def __init__(
        self,
        cell_state_dims: int,
        spatial_dims: int,
        hidden_dims: int,
        key,
        max_volume_difference: float = 0.2,
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

            max_volume_difference:

                The maximum volume difference of the daughter cells, expressed
                in percentage of the parent volume. Controls the asymmetry of
                divisions by limiting the range of the predicted `volume_ratio`
                to `[0.5 - d/2, 0.5 + d/2]` (where `d` is
                `max_volume_difference`). A value of 0.0 will lead to symmetric
                divisions, a value of 1.0 does not restrict the size difference
                at all.
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
        self.max_volume_difference = float(max_volume_difference)

    def __call__(self, cell_state: jax.Array):
        x = cell_state
        for layer in self.layers:
            x = layer(x)
        state_ratio = jax.nn.sigmoid(x[: self.cell_state_dims])
        volume_ratio = (0.5 - self.max_volume_difference / 2) + jax.nn.sigmoid(
            x[self.cell_state_dims]
        ) * self.max_volume_difference
        division_plane = x[self.cell_state_dims + 1 :]
        division_plane = division_plane / jnp.clip(
            jnp.linalg.norm(division_plane), min=1e-10
        )
        return state_ratio, volume_ratio, division_plane


class MechanicsModel(eqx.Module):
    morse_well_width: float
    morse_well_depth: float

    def __init__(self, morse_well_width, morse_well_depth):
        self.morse_well_width = float(morse_well_width)
        self.morse_well_depth = float(morse_well_depth)

    def __call__(self, cells: Cell, extended_attributes: bool = False):
        # mask out inactive cells
        force = morse_force(
            cells.position,
            cells.radius,
            cells.active,
            well_width=self.morse_well_width,
            well_depth=self.morse_well_depth,
        )
        return cells.replace(mechanical_force=force)


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


class DiffusionModel(eqx.Module):
    diffusion_coefs: jax.Array
    degradation_rates: jax.Array

    def __init__(self, diffusion_coefs, degradation_rates):
        self.diffusion_coefs = diffusion_coefs
        self.degradation_rates = degradation_rates

    def __call__(self, cells: Cell):
        concentrations = steady_state_concentrations(
            cells.position,
            cells.radius,
            cells.secretion,
            self.diffusion_coefs,
            self.degradation_rates,
            cells.active,
        )
        return cells.replace(concentration=concentrations)


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


class Model(eqx.Module):
    max_num_cells: int
    cell_state_dims: int
    num_molecules: int
    delta_t: float
    initial_cell_states: jax.Array
    react_model: ReactModel
    motility_model: MotilityModel
    split_prob_model: SplitProbModel
    split_model: SplitModel
    mechanics_model: MechanicsModel
    secretion_model: SecretionModel
    diffusion_model: DiffusionModel
    sensation_model: SensationModel

    def __init__(
        self,
        max_num_cells,
        cell_state_dims,
        num_molecules,
        delta_t,
        react_model,
        motility_model,
        split_prob_model,
        split_model,
        mechanics_model,
        secretion_model,
        diffusion_model,
        sensation_model,
        key,
    ):
        self.max_num_cells = max_num_cells
        self.cell_state_dims = cell_state_dims
        self.num_molecules = num_molecules
        self.delta_t = float(delta_t)
        self.initial_cell_states = jax.random.uniform(
            key=key, shape=(max_num_cells, cell_state_dims)
        )
        self.react_model = react_model
        self.motility_model = motility_model
        self.split_prob_model = split_prob_model
        self.split_model = split_model
        self.mechanics_model = mechanics_model
        self.secretion_model = secretion_model
        self.diffusion_model = diffusion_model
        self.sensation_model = sensation_model

    def initialize_cells(self, key, extended_attributes=False):
        empty = jnp.zeros((), dtype=jnp.float32)

        if extended_attributes:
            extended_attrs = {
                "volume_ratio": jnp.zeros((self.max_num_cells,), dtype=jnp.float32),
                "division_plane": jnp.zeros(
                    (self.max_num_cells, self.motility_model.spatial_dims),
                    dtype=jnp.float32,
                ),
            }
        else:
            extended_attrs = {}

        cells = Cell(
            position=jnp.zeros(
                (self.max_num_cells, self.motility_model.spatial_dims),
                dtype=jnp.float32,
            ),
            radius=jnp.ones((self.max_num_cells,), dtype=jnp.float32),
            state=self.initial_cell_states,
            secretion=jnp.zeros(
                (self.max_num_cells, self.num_molecules), dtype=jnp.float32
            ),
            concentration=jnp.zeros(
                (self.max_num_cells, self.num_molecules), dtype=jnp.float32
            ),
            # initially, only one cell is active
            parent=(-jnp.ones((self.max_num_cells,), dtype=jnp.int16)).at[0].set(0),
            # filled in below
            p_split=empty,
            # filled in below
            split=empty,
            motility_force=jnp.zeros(
                (self.max_num_cells, self.motility_model.spatial_dims),
                dtype=jnp.float32,
            ),
            mechanical_force=jnp.zeros(
                (self.max_num_cells, self.motility_model.spatial_dims),
                dtype=jnp.float32,
            ),
            **extended_attrs,
        )

        # initial positions
        cells = self.motility_model(cells, extended_attributes)

        # initial split decisions
        cells = self.split_prob_model(cells, key, extended_attributes)

        return cells

    def partition(self):
        """Partition the model into (parameters, static)."""
        return eqx.partition(self, eqx.is_inexact_array)

    def hyperparameters(self):
        """Return a dictionary of hyperparameters needed to create a similar model."""
        return {
            "max_num_cells": self.max_num_cells,
            "cell_state_dims": self.cell_state_dims,
            "num_molecules": self.num_molecules,
            "delta_t": self.delta_t,
        }
