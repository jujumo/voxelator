import numpy as np
from skimage import measure
from stl import mesh

bound = np.pi * 3
samples = 80
padding = 1
r = np.linspace(start=-bound, stop=bound, num=samples)
x, y, z = np.meshgrid(r, r, r)
# voxel_grid = np.cos(xx) + np.cos(yy) + np.cos(zz)
voxel_grid = np.sin(x) * np.cos(y) +\
             np.sin(y) * np.cos(z) +\
             np.sin(z) * np.cos(x)
voxel_grid = np.abs(voxel_grid)
voxel_grid = np.pad(voxel_grid, padding, 'constant', constant_values=1)
# Use marching cubes to obtain the surface mesh
vertices, faces, normals, values = measure.marching_cubes(voxel_grid, level=0.2)
maillage = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        maillage.vectors[i][j] = vertices[f[j], :]

maillage.save('gyroid 3x3x3.stl')

if False:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(vertices[faces])
    mesh.set_edgecolor('k')

    # Display resulting triangular mesh using Matplotlib. This can also be done
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, samples + 2 * padding)  # a = 6 (times two for 2nd ellipsoid)
    ax.set_ylim(0, samples + 2 * padding)  # b = 10
    ax.set_zlim(0, samples + 2 * padding)  # c = 16

    plt.tight_layout()
    plt.show()
