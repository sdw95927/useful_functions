# Refer to https://stackoverflow.com/questions/6030098/how-to-display-a-3d-plot-of-a-3d-array-isosurface-with-mplot3d-or-similar
mask_3d = np.transpose(np.array(mask_all_aligned), (1, 2, 0))
print(mask_3d.shape)  # [X, Y, Z]
_values = mask_3d[:50, :50, :]



###################### Colored with z axis ############################
import numpy as np
from numpy import sin, cos, pi
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

vmax = 900

verts, faces, normals, vs = measure.marching_cubes(_values, 0, spacing=(0.1, 0.1, 0.1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:,1], faces, Z=verts[:, 2], # color=np.array([cm.jet(_/vmax) for _ in vs]),  # Doesn't work
                cmap='Spectral', lw=1)
plt.show()


 ###################### Colored with values ############################
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.tri._triangulation import Triangulation

verts, faces, normals, vs = measure.marching_cubes(_values, 0, spacing=(1, 1, 1))
colors = np.max(vs[faces], axis=1)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
polyc = Poly3DCollection(verts[faces], facecolors=[cm.jet(_/vmax) for _ in colors], alpha=0.5, rasterized=True)
# polyc.set_array(vs)
# polyc.set_clim(1, 900)
# Appearance settings
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False) 

ax.add_collection(polyc)
ax.set_xlim(0, 50)
ax.set_ylim(0, 50)
ax.set_zlim(0, 10) 
plt.tight_layout()
plt.show()
