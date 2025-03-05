import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


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


def display_open3d_window(mesh_open3d):
    o3d.visualization.draw_geometries([mesh_open3d], window_name='Open3D')


def display_open3d_plotly(mesh_open3d):
    o3d.visualization.draw_plotly([mesh_open3d])

