from __future__ import division

import numpy

from reikna.helpers import product


class UniformGrid:

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
        self.modes = self.size
        self.V = product(self.box)
