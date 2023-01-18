import numpy as np
from skimage import measure
from stl import mesh
import argparse
import logging
import os.path as path


bound = np.pi * 2
samples = 80
padding = 1
r = np.linspace(start=-np.pi, stop=np.pi, num=samples)
x, y, z = np.meshgrid(r, r, r)
# voxel_grid = np.cos(xx) + np.cos(yy) + np.cos(zz)
voxel_grid = np.sin(x) * np.cos(y) +\
             np.sin(y) * np.cos(z) +\
             np.sin(z) * np.cos(x)

voxel_grid = np.abs(voxel_grid)
voxel_grid = np.pad(voxel_grid, padding, 'constant', constant_values=10)

np.save('gyroid.npy', voxel_grid)
