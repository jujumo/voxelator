import numpy as np
from voxel2stl import voxel2stl_file
import os.path as path
from rich.progress import track

voxel_size = 80, 80, 120
radius = voxel_size[0] // 2 - 5
cube_sizes = [3, 5, 7, 7, 9]
voxel_filepath = '../samples/flyingcubes.npy'
stl_filepath = path.splitext(voxel_filepath)[0] + '.stl'


voxel_grid = np.ones(voxel_size, dtype=float) * 1
for i in track(range(3000)):
    size = np.random.choice(cube_sizes)
    size2 = size // 2
    # x_center = np.random.randint(low=0+size2, high=voxel_size[0]-size2)
    # y_center = np.random.randint(low=0+size2, high=voxel_size[1]-size2)
    z_center = np.random.randint(low=0+size2, high=voxel_size[2]-size2)
    theta = np.random.random() * 2 * np.pi

    x_center = int(np.sin(theta) * radius + voxel_size[0]/2)
    y_center = int(np.cos(theta) * radius + voxel_size[1]/2)
    voxel_grid[x_center-size2: x_center+size2, y_center - size2: y_center + size2, z_center-size2: z_center+size2] = -1.

np.save(voxel_filepath, voxel_grid)

voxel2stl_file(voxel_filepath=voxel_filepath,
               stl_filepath=stl_filepath,
               level_value=0.0,
               padding=(10, 1))