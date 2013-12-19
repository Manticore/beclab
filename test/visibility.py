from __future__ import print_function, division

import numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
import reikna.cluda as cluda
from reikna.cluda import Module

from beclab.integrator import (
    FixedStepIntegrator, Wiener, Drift, Diffusion,
    SSCDStepper, CDStepper, RK4IPStepper, RK46NLStepper)
from beclab.modules import get_drift, get_diffusion
from beclab.grid import UniformGrid, box_3D
from beclab.beam_splitter import BeamSplitter
import beclab.constants as const


class PsiCollector:

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


def run_test(thr, label, stepper_cls, no_losses=False, wigner=False):

    print()
    print(
        "*** Running " + label + ", wigner=" + str(wigner) +
        ", no_losses=" + str(no_losses) + " test")
    print()

    # Simulation parameters
    lattice_size = (8, 8, 64) # spatial lattice points
    paths = 16 if wigner else 1 # simulation paths
    interval = 0.12 # time interval
    samples = 200 # how many samples to take during simulation
    steps = samples * 50 # number of time steps (should be multiple of samples)
    gamma = 0.0 if no_losses else 0.2
    f_detuning = -37
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
    mu = const.muTF3D(freqs, N, states[0])
    zz, yy, xx = numpy.meshgrid(*grid.xs, indexing="ij")
    V = states[0].m / 2 * sum(
        (2 * numpy.pi * f * coords) ** 2 for f, coords in zip(freqs, [zz, yy, xx]))
    psi_TF = numpy.sqrt((mu - V).clip(0) / scattering[0, 0])
    # renormalize to account for coarse grids
    N0 = (numpy.abs(psi_TF) ** 2).sum() * grid.dV
    psi_TF *= numpy.sqrt(N / N0)
    axial_n_max = ((numpy.abs(psi_TF) ** 2).sum((0, 1)) * grid.dxs[0] * grid.dxs[1]).max()

    # Two-component state
    psi0 = numpy.empty((2, paths) + grid.shape, state_dtype)
    psi0[0] = psi_TF
    psi0[1] = 0

    # Initial noise
    if wigner:
        random_normals = (
            rng.normal(size=(2, paths, lattice_size)) +
            1j * rng.normal(size=(2, paths, lattice_size))) / 2
        fft_scale = numpy.sqrt(grid.dV / grid.size)
        psi0 = numpy.fft.ifft(random_normals) / fft_scale + psi0

    psi_gpu = thr.to_device(psi0.astype(state_dtype))

    bs = BeamSplitter(thr, psi_gpu, 0, 1, f_detuning=f_detuning, f_rabi=f_rabi)
    bs(psi_gpu, 0, numpy.pi / 2)

    # Prepare integrator components
    drift = get_drift(state_dtype, grid, states, freqs, scattering, losses, wigner=wigner)
    if wigner:
        diffusion = get_diffusion(state_dtype, grid, 2, losses) if wigner else None
    else:
        diffusion = None

    stepper = stepper_cls(
        grid.shape, grid.box, drift,
        kinetic_coeff=const.HBAR / (2 * states[0].m),
        ensembles=paths,
        diffusion=diffusion)

    if wigner:
        wiener = Wiener(stepper.parameter.dW, 1. / grid.dV, seed=rng.rand())
    integrator = FixedStepIntegrator(
        thr, stepper,
        wiener=wiener if wigner else None)

    # Integrate
    psi_temp = thr.empty_like(psi_gpu)
    psi_collector = PsiCollector(thr, grid, psi_temp, bs, wigner=wigner)
    result, errors = integrator(
        psi_gpu, 0, interval, steps, samples=samples, collectors=[psi_collector])

    N_mean = result['N_mean']
    N_err = result['N_err']
    density = result['axial_density']
    print(density.shape)
    N_exact = N * numpy.exp(-gamma * result['time'] * 2)

    suffix = ('_wigner' if wigner else '') + ('_no-losses' if no_losses else '') + '_' + label

    # Plot density
    print((grid.xs[-1][0] * 1e6, grid.xs[-1][-1] * 1e6))
    for comp in (0, 1):
        fig = plt.figure()
        s = fig.add_subplot(111)
        s.imshow(density[:,comp,:].T, interpolation='nearest', origin='lower', aspect='auto',
            extent=(0, interval) + (grid.xs[-1][0] * 1e6, grid.xs[-1][-1] * 1e6),
            vmin=0, vmax=axial_n_max)
        s.set_xlabel('$t$')
        s.set_ylabel('$x$')
        fig.savefig('visibility_density_' + str(comp) + '_' + suffix + '.pdf')

    fig = plt.figure()
    s = fig.add_subplot(111)
    pz = (density[:,1] - density[:,0]) / (density[:,1] + density[:,0])
    s.imshow(pz.T, interpolation='nearest', origin='lower', aspect='auto',
        extent=(0, interval) + (grid.xs[-1][0] * 1e6, grid.xs[-1][-1] * 1e6),
        vmin=-1, vmax=1)
    s.set_xlabel('$t$')
    s.set_ylabel('$x$')
    fig.savefig('visibility_density_Pz_' + suffix + '.pdf')

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


if __name__ == '__main__':

    # Run integration
    api = cluda.ocl_api()
    thr = api.Thread.create()

    tests = [
        #("SS-CD", SSCDStepper),
        #("CD", CDStepper),
        ("RK4IP", RK4IPStepper),
        #("RK46-NL", RK46NLStepper),
    ]

    for label, cls in tests:
        run_test(thr, label, cls, no_losses=True, wigner=False)
        #run_test(thr, label, cls, wigner=False)
        #run_test(thr, label, cls, wigner=True)
