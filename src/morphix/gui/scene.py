import numpy as np
import pygfx as gfx
import spatial_graph as sg

from ..cell import Cell
from .colors import states_to_rgb


class Scene(gfx.Scene):
    """A pygfx scene to show a single time step.

    Args:
        lineage:

            The lineage to show.

        delta_t:

            The time difference between time steps. Used to show the effective
            forces per time step.

        inactive_cell_opacity:

            Set to a value >0 to show inactive cells.
    """

    def __init__(
        self, lineage: Cell, delta_t: float = 1.0, inactive_cell_opacity: float = 0.0
    ):
        super().__init__()

        self.lineage = lineage
        self.num_timesteps = lineage.parent.shape[0]
        self.max_num_cells = lineage.parent.shape[1]

        self.graph = self._create_lineage_graph(lineage)
        self.inactive_cell_opacity = inactive_cell_opacity
        self.delta_t = delta_t
        self.bb_min, self.bb_max = self._compute_bounding_box()

        self.colors = states_to_rgb(np.array(lineage.state), mode="direct")
        self.bg_dark = (0, 0, 0)
        self.bg_light = (1, 1, 1)
        self.highlight_color = (1, 0.3, 0.8)
        self.hover_color = (0.8, 0.7, 0.8)

        ######################
        # add scene elements #
        ######################

        # background color
        self.background = gfx.Background.from_color(self.bg_dark)
        self.dark_bg = True

        # grid below cells
        self.grid = gfx.Grid(
            None,
            gfx.GridMaterial(
                major_step=1,
                minor_step=0.1,
                thickness_space="world",
                major_thickness=0.01,
                minor_thickness=0.001,
                infinite=True,
            ),
            orientation="xz",
        )
        light = gfx.DirectionalLight(color="white", intensity=5)
        light.local.position = (5, 10, 5)

        self.objects = gfx.Scene()
        self.cached_cells = {}
        self.cached_highlights = {}
        self.hovers = None

        self.add(self.background)
        self.add(self.grid)
        self.add(gfx.AmbientLight(intensity=1.0))
        self.add(light)
        self.add(self.objects)

        self.sphere = gfx.sphere_geometry(
            radius=1,
            width_segments=20,
            height_segments=20,
        )

        self.grid.local.y = self.bb_min[1] - 1.0
        self.highlight = None
        self.hover_nodes = []
        self.hover_distances = []
        self.set_timestep(0)

    @property
    def bounding_box(self):
        return self.bb_min, self.bb_max

    def set_timestep(self, t):
        self.t = t
        self.hover_nodes = []
        self._update_objects()

    def set_highlight(self, index=None):
        """Highlight a cell in the current time step by its index."""
        self.highlight = index
        self._update_objects()

    def set_hover(self, origin, direction):
        hover_pos = np.array([self.t, *origin], dtype="float32")
        hover_dir = np.array([0, *direction], dtype="float32")
        nodes, distances = self.graph.query_nearest_nodes(
            hover_pos, hover_dir, k=100, return_distances=True
        )
        in_timestep = self.graph.node_attrs[nodes].position[:, 0] == self.t
        self.hover_nodes = nodes[in_timestep]
        self.hover_distances = distances[in_timestep]
        self._update_objects()

    def toggle_bg(self):
        self.dark_bg = not self.dark_bg

        assert isinstance(self.background.material, gfx.materials.BackgroundMaterial)
        if self.dark_bg:
            self.background.material.set_colors(self.bg_dark)
        else:
            self.background.material.set_colors(self.bg_light)

    def _create_lineage_graph(self, lineage: Cell):
        graph = sg.SpatialGraph(
            ndims=4,
            node_dtype="uint64",
            node_attr_dtypes={"position": "float32[4]", "radius": "float32"},
            position_attr="position",
            directed=True,
        )

        for t in range(self.num_timesteps):
            active = lineage.active[t]
            positions = lineage.position[t, active]
            positions = np.insert(positions, 0, t, axis=1)  # prepend time
            radii = np.array(lineage.radius[t, active])
            nodes = np.arange(
                t * self.max_num_cells,
                t * self.max_num_cells + active.sum(),
                dtype="uint64",
            )
            graph.add_nodes(nodes, position=positions, radius=radii)

        return graph

    def _update_objects(self):
        self.objects.clear()
        self.objects.add(*self._get_cells())
        self.objects.add(*self._get_highlights())
        self.objects.add(*self._get_hovers())

    def _compute_bounding_box(self):
        active = self.lineage.active
        mins = (
            self.lineage.position[active].reshape(-1, 3)
            - self.lineage.radius[active].reshape(-1, 1)
        ).min(axis=0)
        maxs = (
            self.lineage.position[active].reshape(-1, 3)
            + self.lineage.radius[active].reshape(-1, 1)
        ).max(axis=0)
        return mins, maxs

    def _get_cells(self):
        t = self.t
        if t in self.cached_cells:
            return self.cached_cells[t]

        position = self.lineage.position[t]
        radii = self.lineage.radius[t]
        parents = self.lineage.parent[t]
        active = self.lineage.active[t]
        num_cells = position.shape[0]

        cell_meshes = []
        parent_lines = []
        motility_force_lines = []
        mechanical_force_lines = []

        # draw each cell
        for i in range(num_cells):
            color = self.colors[t * num_cells + i]
            material = gfx.MeshPhongMaterial(
                color=color if active[i] else (1.0, 1.0, 1.0),
                opacity=0.8 if active[i] else self.inactive_cell_opacity,
                alpha_mode="blend",
                shininess=100,
                side="front",
            )
            mesh = gfx.Mesh(self.sphere, material)
            mesh.local.position = position[i]
            s = float(radii[i])
            mesh.local.scale = (s, s, s)
            mesh.render_order = 1 if active[i] else 2
            cell_meshes.append(mesh)

        # draw lines from each cell to its parent in previous timestep
        if t > 0:
            prev_position = np.asarray(self.lineage.position[t - 1], dtype=np.float32)
            segs = []
            for i in range(num_cells):
                parent_index = int(parents[i])
                if parent_index >= 0 and parent_index < prev_position.shape[0]:
                    child = position[i]
                    parent = prev_position[parent_index]
                    segs.append(child)
                    segs.append(parent)
                    # interrupt line
                    segs.append([np.nan, np.nan, np.nan])
            if segs:
                segs = np.asarray(segs, dtype=np.float32)
                geometry = gfx.Geometry()
                buf = gfx.Buffer(segs)
                geometry.positions = buf
                material = gfx.LineMaterial(
                    color=(0.8, 0.2, 0.2), thickness=2.0, opacity=0.5
                )
                lines = gfx.Line(geometry, material)
                parent_lines.append(lines)

            # draw force lines
            if self.lineage.motility_force is not None:
                motility_segs = []
                mechanical_segs = []
                for i in range(num_cells):
                    if not active[i]:
                        continue
                    motility = self.lineage.motility_force[t, i] * self.delta_t
                    mechanical = self.lineage.mechanical_force[t, i] * self.delta_t
                    parent_index = int(parents[i])
                    origin = prev_position[parent_index]
                    motility_segs.append(origin)
                    motility_segs.append(origin + motility)
                    motility_segs.append([np.nan, np.nan, np.nan])
                    mechanical_segs.append(origin)
                    mechanical_segs.append(origin + mechanical)
                    mechanical_segs.append([np.nan, np.nan, np.nan])
                motility_segs = np.asarray(motility_segs, dtype=np.float32)
                mechanical_segs = np.asarray(mechanical_segs, dtype=np.float32)
                geometry = gfx.Geometry()
                buf = gfx.Buffer(motility_segs)
                geometry.positions = buf
                material = gfx.LineMaterial(
                    color=(0.2, 0.8, 0.2), thickness=2.0, opacity=0.5
                )
                lines = gfx.Line(geometry, material)
                motility_force_lines.append(lines)
                geometry = gfx.Geometry()
                buf = gfx.Buffer(mechanical_segs)
                geometry.positions = buf
                material = gfx.LineMaterial(color=(0.2, 0.2, 0.8), thickness=2.0)
                lines = gfx.Line(geometry, material)
                mechanical_force_lines.append(lines)

        objects = (
            cell_meshes + parent_lines + motility_force_lines + mechanical_force_lines
        )
        self.cached_cells[t] = objects

        return objects

    def _get_highlights(self):
        t = self.t
        index = self.highlight
        if index is None:
            return []

        if (t, index) in self.cached_highlights:
            return self.cached_highlights[(t, index)]

        highlights = []
        material = gfx.materials.PointsGaussianBlobMaterial(
            color=self.highlight_color,
            size_mode="vertex",
            size_space="world",
            opacity=0.9,
            alpha_mode="blend",
            depth_write=False,
        )
        geometry = gfx.geometries.Geometry(
            positions=np.array([self.lineage.position[t, index]]),
            sizes=np.array([self.lineage.radius[t, index] * 3]),
        )
        points = gfx.objects.Points(geometry, material)
        highlights.append(points)

        self.cached_highlights[(t, index)] = highlights
        return highlights

    def _get_hovers(self):
        num_visible_hovers = 1
        if self.hovers is None:
            # initialize hover objects
            material = gfx.materials.PointsGaussianBlobMaterial(
                color=self.hover_color,
                size_mode="vertex",
                size_space="world",
                opacity=0.9,
                alpha_mode="blend",
                depth_write=False,
            )
            geometry = gfx.geometries.Geometry(
                positions=np.zeros((num_visible_hovers, 3), dtype=np.float32),
                sizes=np.zeros((num_visible_hovers,), dtype=np.float32),
            )
            self.hovers = gfx.objects.Points(geometry, material)

        num_hovers = len(self.hover_nodes)
        if num_hovers == 0:
            return []

        positions = self.graph.node_attrs[self.hover_nodes].position
        radii = self.graph.node_attrs[self.hover_nodes].radius

        # 1 at center, 0 at boundary, negative outside
        closeness = 1.0 - self.hover_distances / radii
        # ~1 at center, 0.5 at boundary, towards 0 outside
        hover_strength = 1.0 / (1.0 + np.exp(-5 * closeness))
        # increase radius by a factor of at most 1.4
        radii *= 1 + 0.4 * hover_strength

        # we only show the closest hover
        assert isinstance(self.hovers.geometry, gfx.geometries.Geometry)
        self.hovers.geometry.positions.set_data(positions[0:num_visible_hovers, 1:])
        self.hovers.geometry.sizes.set_data(2 * radii[0:num_visible_hovers])

        return [self.hovers]
