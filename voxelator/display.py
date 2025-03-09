import numpy as np
import matplotlib.pyplot as plt
# import open3d as o3d
import trimesh
import pyrender


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


def display_trimesh(mesh):
    # Create a pyrender scene

    # Create a pyrender node from the Trimesh object and add it to the scene
    pyrender_mesh = pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces))

    # Wrap the pyrender mesh in a pyrender node
    mesh_node = pyrender.Node(mesh=pyrender_mesh)
    scene = pyrender.Scene()
    scene.add_node(mesh_node )

    # Set up a simple viewer
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True)


def display_open3d_window(mesh_open3d):
    o3d.visualization.draw_geometries([mesh_open3d], window_name='Open3D')


def display_open3d_plotly(mesh_open3d):
    o3d.visualization.draw_plotly([mesh_open3d])

