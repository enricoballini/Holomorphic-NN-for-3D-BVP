""" """

import numpy as np
import cadquery as cq
import utils_data_and_folders

utils_data_and_folders.setup_directories()

L = 1.0
np.savetxt("./data/L", np.array([L]))

height = L
width = L
thickness = L


result = cq.Workplane("XY").box(height, width, thickness)
result = result.translate((0.5, 0.5, 0.5))

cq.exporters.export(result, "geometry.step")

# for gui: >> CQ-editor
