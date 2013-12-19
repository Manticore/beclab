from __future__ import division

import numpy

from reikna.helpers import product

from beclab.constants import muTF3D


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
        self.V = product(self.box)


def box_3D(N, freqs, state, border=1.2):
    mu = muTF3D(freqs, N, state)
    diameter = lambda f: 2.0 * border * numpy.sqrt(2.0 * mu / (state.m * (2 * numpy.pi * f) ** 2))
    return tuple(diameter(f) for f in freqs)
