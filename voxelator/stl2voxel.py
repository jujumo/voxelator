import numpy as np
import os.path as path
import open3d as o3d
import matplotlib.pyplot as plt


def stl2voxel(
        stl_filepath: str,
        voxel_filepath: str
):

    stl_mesh = o3d.io.read_triangle_mesh(stl_filepath)
    stl_mesh = o3d.t.geometry.TriangleMesh.from_legacy(stl_mesh)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(stl_mesh)  # we do not need the geometry ID for mesh

    min_bound = stl_mesh.vertex['positions'].min(0).numpy()
    max_bound = stl_mesh.vertex['positions'].max(0).numpy()

    xyz_range = np.linspace(min_bound-10, max_bound+10, num=128)

    # query_points is a [32,32,32,3] array ..
    query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)

    # signed distance is an array
    signed_distance = scene.compute_signed_distance(query_points)
    signed_distance = signed_distance.numpy()

    # print(signed_distance.shape)
    # We can visualize a slice of the distance field directly with matplotlib
    plt.imshow(signed_distance[:, :, 10])
    plt.show()

    np.save(voxel_filepath, signed_distance)


def stl2voxel_files(
        stl_filepath: str,
        voxel_filepath: str
):
    stl_mesh = o3d.io.read_triangle_mesh(stl_filepath)