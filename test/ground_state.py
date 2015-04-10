from __future__ import print_function, division

import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reikna import cluda

from beclab import *


dtype = numpy.complex128
shape = (16, 16, 128)
freqs = (97.6, 97.6, 11.96)
components = [const.rb87_1_minus1, const.rb87_2_1]
N = 55000


def ground_state(thomas_fermi=False, use_cutoff=False):

    api = cluda.ocl_api()
    thr = api.Thread.create()

    potential = HarmonicPotential(freqs)
    scattering = const.scattering_matrix(components, B=const.magical_field_Rb87_1m1_2p1)

    system = System(components, scattering, potential=potential)
    grid = UniformGrid(shape, box_for_tf(system, 0, N))

    if use_cutoff:
        cutoff = WavelengthCutoff.padded(grid, pad=4)
        print("Using", cutoff.get_modes_number(grid), "modes out of", grid.size)
    else:
        cutoff = None

    if thomas_fermi:
        gs_gen = ThomasFermiGroundState(thr, dtype, grid, system, cutoff=cutoff)
        gs = gs_gen([N, 0])
    else:
        gs_gen = ImaginaryTimeGroundState(thr, dtype, grid, system, cutoff=cutoff)
        ax_sampler = DensityIntegralSampler(gs_gen.wfs_meta, axes=(0, 1))
        gs, result, info = gs_gen(
            [N, 0], E_diff=1e-7, E_conv=1e-9, sample_time=1e-4,
            samplers=dict(axial_density=ax_sampler), return_info=True)

    cutoff_suffix = '_cutoff' if use_cutoff else ''
    tf_suffix = '_TF' if thomas_fermi else ''

    final_density = numpy.abs(gs.data.get()[0,0]) ** 2
    final_axial_density = final_density.sum((0, 1)) * grid.dxs[0] * grid.dxs[1]

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.plot(grid.xs[2] * 1e6, final_axial_density / 1e6)
    fig.savefig('ground_state_final' + tf_suffix + cutoff_suffix + '.pdf')

    if not thomas_fermi:
        fig = plt.figure()
        s = fig.add_subplot(111)
        s.imshow(result['axial_density']['mean'][:,0,:].T)
        fig.savefig('ground_state_evolution' + cutoff_suffix + '.pdf')

        # Plot used steps
        ts_start, ts_end, steps_used = map(numpy.array, zip(*info.steps))

        fig = plt.figure()
        s = fig.add_subplot(111)
        s.bar(ts_start, steps_used, width=(ts_end - ts_start))
        s.set_xlabel('$t$')
        s.set_ylabel('steps')
        fig.savefig('ground_state_steps' + cutoff_suffix + '.pdf')


if __name__ == '__main__':
    ground_state(thomas_fermi=True, use_cutoff=False)
    ground_state(thomas_fermi=True, use_cutoff=True)
    ground_state(thomas_fermi=False, use_cutoff=False)
    ground_state(thomas_fermi=False, use_cutoff=True)
