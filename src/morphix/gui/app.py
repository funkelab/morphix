import sys

import PyQt5.QtWidgets

from .lineage_viewer import LineageViewer


def show_lineage(lineage, delta_t=1.0, inactive_cell_opacity=0.0):
    """Show a lineage in the lineage viewer."""
    app = PyQt5.QtWidgets.QApplication(sys.argv)
    win = LineageViewer(lineage, delta_t, inactive_cell_opacity=inactive_cell_opacity)
    win.setWindowTitle("morphix lineage viewer")
    win.resize(1200, 800)
    win.show()
    app.exec_()
