import matplotlib.pyplot as plt
import numpy as np
import skimage
from typing import Tuple, Optional
import trimesh
from PIL import Image, ImageOps


def voxel_to_trimesh(
        voxel_grid: np.ndarray,
        level: float = 0.0,
        scale: float = 1.0,
        gradient_direction='descent',
) -> Tuple:
    """
    Convert a voxel grid to a mesh (vertices, faces, normals)
    It uses marching cubes algorithm to obtain the surface mesh.
    """
    vertices, faces, normals, _ = skimage.measure.marching_cubes(
        voxel_grid,
        level=level,
        gradient_direction=gradient_direction
    )

    # apply scale
    vertices = vertices * scale
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    return mesh


def image_file_to_voxel(
    img_filepath: str
) -> np.ndarray:
    """
    Convert an image file into a voxel grid,
    see image_to_voxel for details
    """
    img = Image.open(img_filepath)
    voxel_grid = image_to_voxel(img)
    return voxel_grid


def image_to_voxel(
        img: Image.Image,
) -> np.ndarray:
    """
    Convert an image  into a voxel grid,
    where the grid is single layer of voxels, each cell containing the pixel value of the image.
    the value is normalized from 0..255 to 0..1
    """
    img = ImageOps.grayscale(img)
    voxel_grid = np.array(img).astype(float) / 255.
    voxel_grid = np.expand_dims(voxel_grid, axis=2)
    return voxel_grid


def elevation_to_voxel(
    elevation_map: np.ndarray,
):
    """
    Convert an elevation array into a voxel grid,
    The elevation_map is a 2-D array. Each element contains the height of the surface.
    """
    max_value = np.max(elevation_map)  # maximum depth
    # vertical axis (depth)
    z = np.arange(max_value)
    # Broadcasting : (H, W, 1) - (1, 1, D)
    voxels = z[None, None, :] - elevation_map[:, :, None]
    return voxels


def trimesh_to_stl(
        stl_filepath: str,
        mesh,
        scale: float = 1.0
):
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices * scale
    mesh.export(stl_filepath)


MARGIN = 1
ODD = 1


def mesh_to_voxel(
    mesh,
    voxel_size: float = 1.0
) -> np.ndarray:
    mesh_ranges = mesh.bounds.ptp(axis=0)
    mesh_center = (mesh.bounds[0] + mesh.bounds[1]) / 2.
    grid_shape = np.ceil(mesh_ranges / voxel_size).astype(int) // 2 * 2 + 2 * MARGIN + ODD  # make sure its odd
    grid_coords = np.indices(grid_shape)[:]
    grid_coords = grid_coords.reshape((3, -1), order='C')  # (3, x, y, z) -> (3, n)
    grid_coords = grid_coords.transpose()  # (3, n) -> (n, 3)
    grid_coords = grid_coords * voxel_size  # scale
    grid_coords = grid_coords + (mesh_center - grid_shape//2 * voxel_size).reshape(1, 3)  # match center to center cell
    grid_coords = grid_coords.astype(np.float32)

    try:
        import open3d as o3d
        # Extract vertices and faces from the trimesh.Mesh
        vertices = mesh.vertices
        faces = mesh.faces

        # Create an Open3D mesh object
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d = o3d.t.geometry.TriangleMesh.from_legacy(mesh_o3d)
        mesh_o3d.compute_vertex_normals()
        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh_o3d)  # we do not need the geometry ID for mesh

        # signed distance is an array
        signed_distance = scene.compute_signed_distance(grid_coords)
        voxel_grid = signed_distance.numpy()
    except ImportError:
        # fall back using trimesh only: very slow
        print('warning: using trimesh.proximity.signed_distance is slow: install open3d !')
        voxel_grid = trimesh.proximity.signed_distance(mesh, grid_coords)

    voxel_grid = voxel_grid.reshape(grid_shape[0:3])
    return voxel_grid

