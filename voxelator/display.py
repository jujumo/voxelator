import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def displays_voxel(
        voxel_grid: np.ndarray,
        level_value: float = 0.1
):
    level_voxels = voxel_grid < level_value
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(level_voxels,
              linewidth=0.5)
    ax.set(xlabel='r', ylabel='g', zlabel='b')
    ax.set_aspect('equal')

    plt.show()


    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    # mesh_grid = Poly3DCollection(vertices[faces])
    # mesh_grid.set_edgecolor('k')
    #
    # # Display resulting triangular mesh using Matplotlib. This can also be done
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.add_collection3d(mesh_grid)
    #
    # ax.set_xlabel("x-axis")
    # ax.set_ylabel("y-axis")
    # ax.set_zlabel("z-axis")
    #
    # ax.set_xlim(0, samples + 2 * padding)  # a = 6 (times two for 2nd ellipsoid)
    # ax.set_ylim(0, samples + 2 * padding)  # b = 10
    # ax.set_zlim(0, samples + 2 * padding)  # c = 16
    #
    # plt.tight_layout()
    # plt.show()
