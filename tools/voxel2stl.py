import numpy as np
from stl import mesh
from skimage import measure
import argparse
import logging
import os.path as path
from rich.progress import track
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(name)s::%(levelname)-8s: %(message)s')


def voxel2stl(
        voxel_grid: np.ndarray,
        level_value: float = 0.0,
        padding: Optional[Tuple[int, float]] = None) -> mesh.Mesh:
    if padding:
        padding_size, padding_value = padding[0], padding[1]
        voxel_grid = np.pad(voxel_grid, padding[0], 'constant', constant_values=padding_value)

    # Use marching cubes to obtain the surface mesh
    vertices, faces, normals, values = measure.marching_cubes(
        voxel_grid,
        level=level_value,
        spacing=(1, 1, 1)
    )
    maillage_stl = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in track(enumerate(faces)):
        for j in range(3):
            maillage_stl.vectors[i][j] = vertices[f[j], :]
    return maillage_stl


def voxel2stl_file(
        voxel_filepath: str,
        stl_filepath: str,
        level_value: float = 0.0,
        padding: Optional[Tuple[int, float]] = None
):
    voxel_grid = np.load(voxel_filepath)
    maillage_stl = voxel2stl(voxel_grid=voxel_grid, level_value=level_value, padding=padding)
    maillage_stl.save(stl_filepath)


class VerbosityParsor(argparse.Action):
    """ accept debug, info, ... or theirs corresponding integer value formatted as string."""

    def __call__(self, parser, namespace, values, option_string=None):
        assert isinstance(values, str)
        try:  # in case it represent an int, directly get it
            values = int(values)
        except ValueError:  # else ask logging to sort it out
            values = logging.getLevelName(values.upper())
        setattr(namespace, self.dest, values)


def voxel2stl_cli():
    try:
        parser = argparse.ArgumentParser(description='Convert numpy 3D array to mesh.')
        parser_verbosity = parser.add_mutually_exclusive_group()
        parser_verbosity.add_argument(
            '-v', '--verbose', nargs='?', default=logging.WARNING, const=logging.INFO, action=VerbosityParsor,
            help='verbosity level (debug, info, warning, critical, ... or int value) [warning]')
        parser_verbosity.add_argument(
            '-q', '--silent', '--quiet', action='store_const', dest='verbose', const=logging.CRITICAL)
        parser.add_argument('-i', '--input', required=True,
                            help='input file path')
        parser.add_argument('-o', '--output',
                            help='output file path [input.stl]')
        parser.add_argument('-l', '--level', type=float, default=0.0,
                            help='level value of the surface [0]')
        parser.add_argument('-p', '--padding', type=float,
                            help='add padding value all around')
        args = parser.parse_args()
        logger.setLevel(args.verbose)

        if not args.output:
            args.output = path.splitext(args.input)[0] + '.stl'

        logger.debug(f'loading voxel grid from : {args.input}')
        voxel2stl_file(voxel_filepath=args.input,
                       stl_filepath=args.output,
                       level_value=args.level,
                       padding=(10, args.padding))

    except Exception as e:
        logger.critical(e)
        if args.verbose <= logging.DEBUG:
            raise


if __name__ == '__main__':
    voxel2stl_cli()
