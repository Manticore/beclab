from __future__ import print_function, division

import itertools

import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
import reikna.cluda as cluda
from reikna.cluda import Module

from beclab.integrator import (
    StopIntegration, Integrator, Wiener, Drift, Diffusion,
    SSCDStepper, CDStepper, RK4IPStepper, RK46NLStepper)
from beclab.modules import get_drift, get_diffusion
from beclab.grid import UniformGrid, box_3D
from beclab.beam_splitter import BeamSplitter
import beclab.constants as const
from beclab.ground_state import get_TF_state


class PsiSampler:

    def __init__(self, thr, grid, psi_temp, beam_splitter, wigner=False):
        self._thr = thr
        self._wigner = wigner
        self._grid = grid
        self._beam_splitter = beam_splitter
        self._psi_temp = psi_temp

    def __call__(self, psi, t):
        self._thr.copy_array(psi, dest=self._psi_temp)
        self._beam_splitter(self._psi_temp, t, numpy.pi / 2)
        psi = self._psi_temp.get()

        # (components, ensembles, x_points)
        density = numpy.abs(psi) ** 2 - (0.5 / self._grid.dV if self._wigner else 0)

        N = density.sum((2, 3, 4)) * self._grid.dV
        N_mean = N.mean(1)
        N_err = N.std(1) / numpy.sqrt(N.shape[1])

        axial_density = density.sum((2, 3)).mean(1) * self._grid.dxs[0] * self._grid.dxs[1]

        return dict(
            psi=psi,
            axial_density=axial_density,
            N_mean=N_mean, N_err=N_err
            )


def run_test(thr, stepper_cls, integration, no_losses=False, wigner=False):

    print()
    print(
        "*** Running " + stepper_cls.abbreviation +
        ", " + integration +
        ", wigner=" + str(wigner) +
        ", no_losses=" + str(no_losses) + " test")
    print()

    # Simulation parameters
    lattice_size = (8, 8, 64) # spatial lattice points
    paths = 16 if wigner else 1 # simulation paths
    interval = 0.12 # time interval
    samples = 200 # how many samples to take during simulation
    steps = samples * 100 # number of time steps (should be multiple of samples)
    gamma = 0.0 if no_losses else 0.2
    f_detuning = 37
    f_rabi = 350
    N = 55000
    state_dtype = numpy.complex128

    rng = numpy.random.RandomState(1234)

    freqs = (97.6, 97.6, 11.96)
    states = [const.Rb87_1_minus1, const.Rb87_2_1]

    get_g = lambda s1, s2: const.get_scattering_constant(
        const.get_interaction_constants(const.B_Rb87_1m1_2p1, s1, s2)[0], s1.m)
    scattering = numpy.array([[get_g(s1, s2) for s2 in states] for s1 in states])
    losses = [
        (gamma, (1, 0)),
        (gamma, (0, 1))]

    grid = UniformGrid(lattice_size, box_3D(N, freqs, states[0]))

    # Initial TF state
    psi = get_TF_state(thr, grid, state_dtype, states, freqs, [N, 0])
    axial_n_max = (
        (numpy.abs(psi.data.get()[0,0]) ** 2).sum((0, 1)) * grid.dxs[0] * grid.dxs[1]).max()

    # Initial noise
    if wigner:
        psi = psi.to_wigner_coherent(paths, seed=rng.randint(0, 2**32-1))

    bs = BeamSplitter(thr, psi.data, 0, 1, f_detuning=f_detuning, f_rabi=f_rabi)

    # Prepare integrator components
    drift = get_drift(state_dtype, grid, states, freqs, scattering, losses, wigner=wigner)
    if wigner:
        diffusion = get_diffusion(state_dtype, grid, 2, losses) if wigner else None
    else:
        diffusion = None

    stepper = stepper_cls(
        grid.shape, grid.box, drift,
        kinetic_coeff=1j * const.HBAR / (2 * states[0].m),
        ensembles=paths,
        diffusion=diffusion)

    if wigner:
        wiener = Wiener(stepper.parameter.dW, 1. / grid.dV, seed=rng.randint(0, 2**32-1))
    integrator = Integrator(
        thr, stepper,
        wiener=wiener if wigner else None)

    # Integrate
    psi_temp = thr.empty_like(psi.data)
    psi_sampler = PsiSampler(thr, grid, psi_temp, bs, wigner=wigner)
    bs(psi.data, 0, numpy.pi / 2)

    if integration == 'fixed':
        result, info = integrator.fixed_step(
            psi.data, 0, interval, steps, samples=samples, samplers=[psi_sampler])
    elif integration == 'adaptive':
        result, info = integrator.adaptive_step(
            psi.data, 0, interval / samples, t_end=interval,
            convergence=dict(N_mean=1e-4), samplers=[psi_sampler])

    N_mean = result['N_mean']
    N_err = result['N_err']
    density = result['axial_density']
    N_exact = N * numpy.exp(-gamma * result['time'] * 2)

    suffix = (
        ('_wigner' if wigner else '') +
        ('_no-losses' if no_losses else '') +
        '_' + stepper_cls.abbreviation +
        '_' + integration)

    # Plot density
    for comp in (0, 1):
        fig = plt.figure()
        s = fig.add_subplot(111)
        s.imshow(density[:,comp,:].T, interpolation='nearest', origin='lower', aspect='auto',
            extent=(0, interval) + (grid.xs[-1][0] * 1e6, grid.xs[-1][-1] * 1e6),
            vmin=0, vmax=axial_n_max)
        s.set_xlabel('$t$')
        s.set_ylabel('$x$')
        fig.savefig('visibility_density_' + str(comp) + suffix + '.pdf')
        plt.close(fig)

    fig = plt.figure()
    s = fig.add_subplot(111)
    pz = (density[:,1] - density[:,0]) / (density[:,1] + density[:,0])
    s.imshow(pz.T, interpolation='nearest', origin='lower', aspect='auto',
        extent=(0, interval) + (grid.xs[-1][0] * 1e6, grid.xs[-1][-1] * 1e6),
        vmin=-1, vmax=1)
    s.set_xlabel('$t$')
    s.set_ylabel('$x$')
    fig.savefig('visibility_density_Pz' + suffix + '.pdf')
    plt.close(fig)

    # Plot population
    fig = plt.figure()
    s = fig.add_subplot(111)
    s.plot(result['time'], N_mean[:,0], 'r-')
    s.plot(result['time'], N_mean[:,1], 'g-')
    s.plot(result['time'], N_mean.sum(1), 'b-')
    if wigner:
        s.plot(result['time'], N_mean.sum(1) + N_err.sum(1), 'b--')
        s.plot(result['time'], N_mean.sum(1) - N_err.sum(1), 'b--')
    s.plot(result['time'], N_exact, 'k--')
    s.set_ylim(0, N)
    s.set_xlabel('$t$')
    s.set_ylabel('$N$')
    fig.savefig('visibility_N' + suffix + '.pdf')
    plt.close(fig)

    # Plot used steps
    ts_start, ts_end, steps_used = map(numpy.array, zip(*info.steps))

    fig = plt.figure()
    s = fig.add_subplot(111)
    s.bar(ts_start, steps_used, width=(ts_end - ts_start))
    s.set_xlabel('$t$')
    s.set_ylabel('steps')
    fig.savefig('visibility_steps' + suffix + '.pdf')
    plt.close(fig)


if __name__ == '__main__':

    # Run integration
    api = cluda.ocl_api()
    thr = api.Thread.create()

    steppers = [
        SSCDStepper,
        CDStepper,
        RK4IPStepper,
        RK46NLStepper,
    ]

    integrations = [
        'fixed',
        'adaptive',
    ]

    for stepper_cls, integration in itertools.product(steppers, integrations):
        run_test(thr, stepper_cls, integration, no_losses=True, wigner=False)
        run_test(thr, stepper_cls, integration, wigner=False)
        if integration == 'fixed':
            run_test(thr, stepper_cls, integration, wigner=True)
