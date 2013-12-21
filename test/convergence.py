from __future__ import print_function, division

import numpy

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
from beclab.ground_state import get_TF_state, to_wigner


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

        SZ2s = ((0.25 * (numpy.abs(psi[0]) ** 2 - numpy.abs(psi[1]) ** 2) ** 2).sum((1,2,3))
                * self._grid.dV)
        SZ2_mean = SZ2s.mean()

        return dict(
            N_mean=N_mean, N_err=N_err,
            SZ2_mean=SZ2_mean,
            )


def run_test(thr, label, stepper_cls, steps):

    print()
    print("*** Running " + label + " test, " + str(steps) + " steps")
    print()

    # Simulation parameters
    lattice_size = (8, 8, 64) # spatial lattice points
    paths = 64 # simulation paths
    interval = 0.1 # time interval
    samples = 1 # how many samples to take during simulation
    #steps = samples * 50 # number of time steps (should be multiple of samples)
    gamma = 0.2
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
    psi_TF = get_TF_state(grid, states[0], freqs, N)

    # Two-component state
    psi0 = numpy.empty((2, paths) + grid.shape, state_dtype)
    psi0[0] = psi_TF
    psi0[1] = 0

    # Initial noise
    psi0 = to_wigner(grid, rng, psi0)

    psi_gpu = thr.to_device(psi0.astype(state_dtype))

    bs = BeamSplitter(thr, psi_gpu, 0, 1, f_detuning=f_detuning, f_rabi=f_rabi)

    # Prepare integrator components
    drift = get_drift(state_dtype, grid, states, freqs, scattering, losses, wigner=True)
    diffusion = get_diffusion(state_dtype, grid, 2, losses)

    stepper = stepper_cls(
        grid.shape, grid.box, drift,
        kinetic_coeff=1j * const.HBAR / (2 * states[0].m),
        ensembles=paths,
        diffusion=diffusion)

    wiener = Wiener(stepper.parameter.dW, 1. / grid.dV, seed=rng.randint(0, 2**32-1))
    integrator = FixedStepIntegrator(
        thr, stepper,
        wiener=wiener,
        profile=True)

    # Integrate
    psi_temp = thr.empty_like(psi_gpu)
    psi_collector = PsiCollector(thr, grid, psi_temp, bs, wigner=True)
    bs(psi_gpu, 0, numpy.pi / 2)
    result, info = integrator(
        psi_gpu, 0, interval, steps, samples=samples, collectors=[psi_collector])

    N_mean = result['N_mean']
    N_exact = N * numpy.exp(-gamma * result['time'] * 2)

    return dict(
        strong_error=info.error_strong,
        N_convergence=info.errors['N_mean'],
        SZ2_convergence=info.errors['SZ2_mean'],
        N_diff=abs(N_mean.sum(1)[-1] - N_exact[-1]) / N_exact[-1],
        time=info.t_integration,
        )


if __name__ == '__main__':

    # Run integration
    api = cluda.ocl_api()
    thr = api.Thread.create()

    tests = [
        ("SS-CD", SSCDStepper),
        ("CD", CDStepper),
        ("RK4IP", RK4IPStepper),
        ("RK46-NL", RK46NLStepper),
    ]

    import pickle
    import os.path
    fname = 'convergence.pickle'

    if not os.path.exists(fname):
        with open(fname, 'wb') as f:
            pickle.dump({}, f, protocol=2)

    step_nums = [1, 2, 3, 5, 7]
    step_nums = sum(([1000 * num * 10 ** pwr for num in step_nums] for pwr in range(3)), [])

    for label, cls in tests:
        for steps in step_nums:
            result = run_test(thr, label, cls, steps)
            print(result)

            with open(fname, 'rb') as f:
                results = pickle.load(f)
            if label not in results:
                results[label] = {}
            results[label][steps] = result
            with open(fname, 'wb') as f:
                pickle.dump(results, f, protocol=2)
