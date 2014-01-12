from __future__ import print_function, division

import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reikna import cluda

from beclab import *


if __name__ == '__main__':

    dtype = numpy.complex128
    shape = (8, 8, 64)
    freqs = (97.6, 97.6, 11.96)
    components = [const.rb87_1_minus1, const.rb87_2_1]
    N = 55000

    api = cluda.ocl_api()
    thr = api.Thread.create()

    potential = HarmonicPotential(dtype, freqs)
    scattering = const.scattering_matrix(components, B=const.magical_field_Rb87_1m1_2p1)
    system = System(components, scattering, potential=potential)
    grid = UniformGrid(shape, box_for_tf(system, 0, N))

    gs_gen = ImaginaryTimeGroundState(thr, dtype, grid, system)
    ax_sampler = Density1DSampler(gs_gen.wfs_meta, theta=numpy.pi / 2)

    gs, result, info = gs_gen(
        [N, 0], E_diff=1e-7, E_conv=1e-9, sample_time=1e-4,
        samplers=dict(axial_density=ax_sampler), return_info=True)

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.plot(grid.xs[2] * 1e6, result['axial_density']['mean'][-1,0])
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
