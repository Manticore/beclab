from __future__ import print_function, division

import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
import reikna.cluda as cluda
from reikna.cluda import Module
from reikna.algorithms import PureParallel
from reikna.core import Computation, Parameter, Annotation, Transformation

from beclab.integrator import (
    Sampler, StopIntegration, Integrator, Wiener, Drift, Diffusion,
    CDIPStepper, CDStepper, RK4IPStepper, RK46NLStepper)
from beclab.modules import get_drift, get_diffusion
from beclab.grid import UniformGrid, box_3D
from beclab.beam_splitter import BeamSplitter
import beclab.constants as const
from beclab.ground_state import get_TF_state

from beclab import WavefunctionSet
from beclab.meters import PopulationMeter, EnergyMeter


def get_multiply(wfs):

    real_dtype = dtypes.real_for(wfs.dtype)

    return PureParallel(
        [
            Parameter('output', Annotation(wfs.data, 'o')),
            Parameter('input', Annotation(wfs.data, 'i'))]
            + [Parameter('coeff' + str(i), Annotation(real_dtype)) for i in range(wfs.components)],
        """
        <%
            all_indices = ', '.join(idxs)
        %>
        %for comp in range(components):
        ${output.ctype} psi_${comp} = ${input.load_idx}(${comp}, ${all_indices});
        ${output.store_idx}(${comp}, ${all_indices},
            ${mul}(psi_${comp}, ${locals()['coeff' + str(comp)]}));
        %endfor
        """,
        guiding_array=wfs.shape[1:],
        render_kwds=dict(
            components=wfs.components,
            mul=functions.mul(wfs.dtype, real_dtype)))


class EnergySampler(Sampler):

    def __init__(self, energy_meter):
        Sampler.__init__(self, no_stderr=True)
        self._energy = energy_meter

    def __call__(self, psi, t):
        return self._energy(psi)


class AxialViewSampler(Sampler):

    def __init__(self, grid):
        Sampler.__init__(self, no_stderr=True)
        self._grid = grid

    def __call__(self, psi, t):
        psi = psi.get()
        density = numpy.abs(psi) ** 2
        return (density.sum((2, 3)) * self._grid.dxs[0] * self._grid.dxs[1]).transpose(1, 0, 2)


class PropagationStop(Sampler):

    def __init__(self, energy_meter, limit=1e-6):
        Sampler.__init__(self, no_stderr=True)
        self._energy = energy_meter
        self._previous_E = None
        self._limit = limit

    def __call__(self, psi, t):
        E = self._energy(psi)

        if self._previous_E is not None:
            if abs(E[0] - self._previous_E[0]) / E[0] < self._limit:
                raise StopIntegration(E)

        self._previous_E = E
        return E


class PsiFilter:

    def __init__(self, thr, wfs, target_Ns, population_meter):
        self._population = population_meter
        self._target_Ns = numpy.array(target_Ns)
        self._multiply = get_multiply(wfs).compile(thr)

    def __call__(self, psi, t):
        Ns = self._population(psi).mean(1)
        self._multiply(psi, psi, *(numpy.sqrt(self._target_Ns / Ns)))


if __name__ == '__main__':

    dtype = numpy.complex128
    shape = (8, 8, 64)
    freqs = (97.6, 97.6, 11.96)
    state = const.Rb87_1_minus1
    N = 55000

    api = cluda.ocl_api()
    thr = api.Thread.create()

    get_g = lambda s1, s2: const.get_scattering_constant(
        const.get_interaction_constants(const.B_Rb87_1m1_2p1, s1, s2)[0], s1.m)
    scattering = numpy.array([[get_g(state, state)]])

    grid = UniformGrid(shape, box_3D(N, freqs, state))

    # Initial TF state
    psi = get_TF_state(thr, grid, dtype, [state], freqs, [N])

    drift = get_drift(dtype, grid, [state], freqs, scattering, imaginary_time=True)

    stepper = RK4IPStepper(
        grid.shape, grid.box, drift,
        kinetic_coeff=const.HBAR / (2 * state.m))
    integrator = Integrator(thr, stepper, verbose=True)

    # Integrate
    energy_meter = EnergyMeter(thr, psi, -const.HBAR ** 2 / state.m, [state], freqs, scattering)
    population_meter = PopulationMeter(thr, psi)
    e_sampler = EnergySampler(energy_meter)
    e_stopper = PropagationStop(energy_meter, limit=1e-9)
    ax_sampler = AxialViewSampler(grid)
    psi_filter = PsiFilter(thr, psi, [N], population_meter)
    result, info = integrator.adaptive_step(
        psi.data, 0, 0.001,
        display=['time', 'E'],
        samplers=dict(E_conv=e_sampler, E=e_stopper, axial_density=ax_sampler),
        filters=[psi_filter],
        convergence=dict(E_conv=1e-9))

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.plot(grid.xs[2], result['axial_density'][-1,0])
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
