from __future__ import division

import numpy

from reikna.helpers import product


class Grid:
    """
    The base class for simulation grids.

    .. py:attribute:: box

        A tuple with simulation box sizes.

    .. py:attribute:: dimensions

        The number of dimensions (the length of ``shape`` or ``box`` tuples).

    .. py:attribute:: dV

        The volume of a grid cell.

    .. py:attribute:: dxs

        A list of grid steps for each dimension.

    .. py:attribute:: shape

        A tuple with the grid shape.

    .. py:attribute:: size

        The total number of grid cells.

    .. py:attribute:: xs

        A list of ``dimensions`` numpy arrays containing grid cell coordinates for each dimension.
    """
    pass


class UniformGrid(Grid):
    """
    Rectangular uniform grid.

    :param shape: a tuple with the grid shape.
    :param box: a tuple with the simulation box size (same length as ``shape``).

    .. py:attribute:: V

        The total volume of the simulation box.
    """

    def __init__(self, shape, box):
        if isinstance(shape, int):
            shape = (shape,)
            box = (box,)

        assert len(shape) in [1, 2, 3]

        self.dimensions = len(shape)
        self.shape = shape
        self.box = box

        # spatial step and grid for every component of shape
        self.dxs = [l / n for l, n in zip(self.box, self.shape)]
        self.xs = [
            numpy.linspace(-l / 2 + dx / 2, l / 2 - dx / 2, n)
            for l, n, dx in zip(self.box, self.shape, self.dxs)]

        # number of cells and cell volume
        self.dV = product(self.dxs)
        self.size = product(self.shape)
        self.V = product(self.box)
