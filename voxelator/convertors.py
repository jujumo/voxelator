import numpy as np
import skimage
from typing import Tuple
import trimesh
# import open3d as o3d
from rich.progress import track


def voxel2trimesh(
        voxel_grid: np.ndarray,
        level: float = 0.0,
        scale: float = 1.0,
        center: bool = True
) -> Tuple:
    # Use marching cubes to obtain the surface mesh
    vertices, faces, normals, _ = skimage.measure.marching_cubes(
        voxel_grid,
        level=level,
        spacing=(1, 1, 1)
    )

    # apply scale
    vertices = vertices * scale
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    if center:
        mesh.apply_translation(-mesh.centroid)  # center mesh to 0.
    return mesh


# def trimesh2o3d(
#         mesh
# ):
#     # Convert to Open3D mesh
#     o3d_mesh = o3d.geometry.TriangleMesh()
#     o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
#     o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
#     # Optionally, compute vertex normals for better visualization
#     o3d_mesh.compute_vertex_normals()
#     return o3d_mesh


def trimesh2stl(
        stl_filepath: str,
        mesh,
        scale: float = 1.0
):
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices * scale
    mesh.export(stl_filepath)


# def stl2voxel(
#         stl_filepath: str
# ) -> np.ndarray:
#     stl_mesh = o3d.io.read_triangle_mesh(stl_filepath)
#     stl_mesh = o3d.t.geometry.TriangleMesh.from_legacy(stl_mesh)
#
#     scene = o3d.t.geometry.RaycastingScene()
#     _ = scene.add_triangles(stl_mesh)  # we do not need the geometry ID for mesh
#
#     min_bound = stl_mesh.vertex['positions'].min(0).numpy()
#     max_bound = stl_mesh.vertex['positions'].max(0).numpy()
#
#     xyz_range = np.linspace(min_bound-10, max_bound+10, num=128)
#
#     # query_points is a [32,32,32,3] array ..
#     query_points = np.stack(np.meshgrid(*xyz_range.T), axis=-1).astype(np.float32)
#
#     # signed distance is an array
#     signed_distance = scene.compute_signed_distance(query_points)
#     signed_distance = signed_distance.numpy()
#     return signed_distance
