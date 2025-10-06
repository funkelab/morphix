import sys

import numpy as np
import umap

try:
    import pygfx as gfx
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QApplication,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSlider,
        QVBoxLayout,
        QWidget,
    )
    from rendercanvas.auto import RenderCanvas
except ImportError as e:
    raise ImportError(
        "This module requires additional dependencies to be installed. "
        "Please install 'morphix[gui]'."
    ) from e


class LineageViewer(QWidget):
    def __init__(self, lineage, parent=None):
        super().__init__(parent)
        self.lineage = lineage
        self.n_timesteps = lineage.position.shape[0]
        self.current_t = 0
        self.playing = False

        # compute colors from UMAP of cell state
        print("Computing colors from state...")
        colors = umap.UMAP(
            n_components=3,
            random_state=1912,
        ).fit_transform(lineage.state.reshape(-1, lineage.state.shape[-1]))
        colors_min = colors.min(axis=0)
        colors_max = colors.max(axis=0)
        self.colors = (colors - colors_min) / (colors_max - colors_min)

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.canvas = RenderCanvas()
        layout.addWidget(self.canvas, stretch=1)

        self.play_button = QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.clicked.connect(self.toggle_play)

        self.t_label = QLabel(f"t = 0 / {self.n_timesteps - 1}")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.n_timesteps - 1))
        self.slider.setValue(0)
        self.slider.setSingleStep(1)
        self.slider.valueChanged.connect(self.on_slider_changed)

        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.t_label)
        controls.addWidget(self.slider, stretch=1)
        layout.addLayout(controls)

        self.timer = QTimer()
        self.timer.setInterval(200)  # ms per frame
        self.timer.timeout.connect(self.advance_frame)

        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)
        self.camera = gfx.PerspectiveCamera(60, 16 / 9)
        self.camera.local.position = (0, 0, 10)
        self.camera.look_at((0, 0, 0))
        self.controller = gfx.FlyController(self.camera, register_events=self.renderer)
        self.controller.enable_keys = True
        self.controller.controls["r"] = ("move", "repeat", (0.0, 1.0, 0.0))
        self.controller.controls["f"] = ("move", "repeat", (0.0, -1.0, 0.0))

        light = gfx.DirectionalLight(color=(1, 1, 1), intensity=1.0)
        light.local.position = (5, 10, 5)

        self.objects = gfx.Scene()
        self.cached_objects = {}

        self.scene = gfx.Scene()
        self.scene.add(gfx.AmbientLight(intensity=1.0))
        self.scene.add(light)
        self.scene.add(self.objects)

        self.sphere = gfx.sphere_geometry(
            radius=1.0,
            width_segments=20,
            height_segments=20,
        )

        self.reset_camera()
        self.update_scene_for_t(0)
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def compute_scene_bounds(self):
        mins = (
            self.lineage.position.reshape(-1, 3) - self.lineage.radius.reshape(-1, 1)
        ).min(axis=0)
        maxs = (
            self.lineage.position.reshape(-1, 3) + self.lineage.radius.reshape(-1, 1)
        ).max(axis=0)
        return mins, maxs

    def reset_camera(self, margin=1.2):
        """Position the camera so all objects are visible."""
        min_corner, max_corner = self.compute_scene_bounds()
        center = (min_corner + max_corner) / 2.0
        extent = max_corner - min_corner
        radius = np.linalg.norm(extent) / 2.0 * margin

        if isinstance(self.camera, gfx.PerspectiveCamera):
            # Compute distance from center based on FOV
            fov = np.deg2rad(self.camera.fov)
            distance = radius / np.sin(fov / 2.0)
            self.camera.local.position = (center[0], center[1], center[2] + distance)
            self.camera.look_at(center)
        elif isinstance(self.camera, gfx.OrthographicCamera):
            self.camera.width = extent[0] * margin
            self.camera.height = extent[1] * margin
            self.camera.local.position = (center[0], center[1], center[2] + radius)
            self.camera.look_at(center)

    def toggle_play(self, checked):
        self.playing = bool(checked)
        if self.playing:
            self.play_button.setText("Pause")
            self.timer.start()
        else:
            self.play_button.setText("Play")
            self.timer.stop()

    def advance_frame(self):
        t = (self.current_t + 1) % self.n_timesteps
        self.slider.setValue(t)

    def on_slider_changed(self, t):
        t = int(t)
        if t == self.current_t:
            return
        self.current_t = t
        self.t_label.setText(f"t = {self.current_t} / {self.n_timesteps - 1}")
        self.update_scene_for_t(self.current_t)
        self.canvas.update()

    def clear_scene_objects(self):
        self.objects.clear()

    def update_scene_for_t(self, t):
        self.clear_scene_objects()
        self.objects.add(*self.get_objects(t))

    def get_objects(self, t):
        if t in self.cached_objects:
            return self.cached_objects[t]

        position = self.lineage.position[t]
        radii = self.lineage.radius[t]
        parents = self.lineage.parent[t]
        active = parents >= 0
        num_cells = position.shape[0]

        cell_meshes = []
        parent_lines = []

        # draw each cell
        for i in range(num_cells):
            color = self.colors[t * num_cells + i]
            material = gfx.MeshPhongMaterial(
                color=color if active[i] else (1.0, 1.0, 1.0),
                opacity=0.8 if active[i] else 0.1,
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
                material = gfx.LineMaterial(color=(0.8, 0.2, 0.2), thickness=2.0)
                lines = gfx.Line(geometry, material)
                parent_lines.append(lines)

        objects = cell_meshes + parent_lines
        self.cached_objects[t] = objects

        return objects


def show_lineage(lineage):
    app = QApplication(sys.argv)
    win = LineageViewer(lineage)
    win.setWindowTitle("morphix lineage viewer")
    win.resize(1200, 800)
    win.show()
    app.exec_()
