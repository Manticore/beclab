from __future__ import print_function, division

import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reikna import cluda

from beclab.integrator import Sampler
import beclab.constants as const
from beclab.ground_state import it_ground_state
from beclab.bec import System, box_for_tf, HarmonicPotential
from beclab.grid import UniformGrid



class AxialViewSampler(Sampler):

    def __init__(self, grid):
        Sampler.__init__(self, no_stderr=True)
        self._grid = grid

    def __call__(self, psi, t):
        psi = psi.get()
        density = numpy.abs(psi) ** 2
        return (density.sum((2, 3)) * self._grid.dxs[0] * self._grid.dxs[1]).transpose(1, 0, 2)


if __name__ == '__main__':

    dtype = numpy.complex128
    shape = (8, 8, 64)
    freqs = (97.6, 97.6, 11.96)
    comp = const.rb87_1_minus1
    N = 55000

    api = cluda.ocl_api()
    thr = api.Thread.create()

    scattering = const.scattering_matrix([comp])
    potential = HarmonicPotential(freqs)
    system = System([comp], scattering, potential=potential)
    box = box_for_tf(system, 0, N)
    grid = UniformGrid(shape, box)
    ax_sampler = AxialViewSampler(grid)

    gs, result, info = it_ground_state(
        thr, grid, dtype, system, [N], E_diff=1e-7, E_conv=1e-9, sample_time=1e-4,
        samplers=dict(axial_density=ax_sampler), return_info=True)

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.plot(grid.xs[2] * 1e6, result['axial_density'][-1,0])
    fig.savefig('ground_state_final.pdf')

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.imshow(result['axial_density'][:,0,:].T)
    fig.savefig('ground_state_evolution.pdf')

    # Plot used steps
    ts_start, ts_end, steps_used = map(numpy.array, zip(*info.steps))

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.bar(ts_start, steps_used, width=(ts_end - ts_start))
    s.set_xlabel('$t$')
    s.set_ylabel('steps')
    fig.savefig('ground_state_steps.pdf')
