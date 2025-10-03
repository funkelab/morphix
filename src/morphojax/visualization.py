import sys

import numpy as np

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
    from wgpu.gui.qt import WgpuCanvas
except ImportError as e:
    raise ImportError(
        "This module requires additional dependencies to be installed. "
        "Please install 'morphojax[gui]'."
    ) from e


class LineageViewer(QWidget):
    def __init__(self, lineage, parent=None):
        super().__init__(parent)
        self.lineage = lineage
        self.n_timesteps = lineage.position.shape[0]
        self.current_t = 0
        self.playing = False

        layout = QVBoxLayout()
        self.setLayout(layout)
        self.canvas = WgpuCanvas()
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
        self.scene = gfx.Scene()
        self.camera = gfx.PerspectiveCamera(60, 16 / 9)
        self.camera.local.position = (0, 0, 10)
        self.camera.look_at((0, 0, 0))
        self.controller = gfx.FlyController(self.camera, register_events=self.renderer)
        self.controller.enable_keys = True

        self.scene.add(gfx.AmbientLight(intensity=1.0))
        light = gfx.DirectionalLight(color=(1, 1, 1), intensity=1.0)
        light.local.position = (5, 10, 5)
        self.scene.add(light)

        self.sphere = gfx.sphere_geometry(
            radius=1.0,
            width_segments=20,
            height_segments=20,
        )

        self.cell_meshes = []
        self.parent_lines = []

        self.update_scene_for_t(0)
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

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
        for m in self.cell_meshes:
            self.scene.remove(m)
        for line in self.parent_lines:
            self.scene.remove(line)
        self.cell_meshes = []
        self.parent_lines = []

    def update_scene_for_t(self, t):
        self.clear_scene_objects()

        position = self.lineage.position[t]
        sizes = self.lineage.size[t]
        parents = self.lineage.parent[t]
        active = parents >= 0
        num_cells = position.shape[0]

        # draw each cell
        for i in range(num_cells):
            material = gfx.MeshPhongMaterial(
                color=(0.8, 0.6, 0.8) if active[i] else (1.0, 1.0, 1.0),
                opacity=0.8 if active[i] else 0.1,
                shininess=100,
                side="front",
            )
            mesh = gfx.Mesh(self.sphere, material)
            mesh.local.position = position[i]
            s = float(sizes[i])
            mesh.local.scale = (s, s, s)
            mesh.render_order = 1 if active[i] else 2
            self.scene.add(mesh)
            self.cell_meshes.append(mesh)

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
                self.scene.add(lines)
                self.parent_lines.append(lines)


def show_lineage(lineage):
    app = QApplication(sys.argv)
    win = LineageViewer(lineage)
    win.setWindowTitle("morphojax lineage viewer")
    win.resize(1200, 800)
    win.show()
    app.exec_()
