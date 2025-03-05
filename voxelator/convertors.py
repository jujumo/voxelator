import numpy as np
import skimage
from typing import Tuple
import open3d as o3d
import stl
from rich.progress import track


def voxel2surface(
        voxel_grid: np.ndarray,
        level: float = 0.0,
        scale: float = 1.0
) -> Tuple:
    # Use marching cubes to obtain the surface mesh
    vertices, faces, normals, _ = skimage.measure.marching_cubes(
        voxel_grid,
        level=level,
        spacing=(1, 1, 1)
    )

    # apply scale
    vertices = vertices * scale
    return vertices, faces, normals


def surface2open3D(
        surface
):
    vertices, faces, normals = surface
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
    # Optionally, compute vertex normals for better visualization
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d


def surface2stl(
        stl_filepath: str,
        surface,
        scale: float = 1.0
):
    vertices, faces, normals = surface
    vertices = scale * vertices
    stl_mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    progress = list(enumerate(faces))
    for i, f in track(progress):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[f[j], :]
    stl_mesh.save(stl_filepath)


def stl2voxel(
        stl_filepath: str
) -> np.ndarray:
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
    return signed_distance
