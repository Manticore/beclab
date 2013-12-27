from __future__ import print_function, division

import numpy

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
import reikna.cluda as cluda
from reikna.cluda import Module

from beclab.integrator import (
    Integrator, Wiener, Drift, Diffusion,
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
        psi_orig = psi.get()
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
            psi=psi_orig,
            N_mean=N_mean, N_err=N_err,
            SZ2_mean=SZ2_mean,
            )


def run_test(thr, stepper_cls, steps):

    print()
    print("*** Running " + stepper_cls.abbreviation + " test, " + str(steps) + " steps")
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
        (5.4e-42 / 6, (3, 0)),
        (8.1e-20 / 4, (0, 2)),
        (1.51e-20 / 2, (1, 1)),
        ]

    grid = UniformGrid(lattice_size, box_3D(N, freqs, states[0]))

    # Initial TF state
    psi = get_TF_state(thr, grid, state_dtype, states, freqs, [N, 0])

    # Initial ground state
    with open('ground_state_8-8-64_1-1-1.pickle', 'rb') as f:
        data = pickle.load(f)

    psi_gs = data['data']
    psi.fill_with(psi_gs)

    # Initial noise
    psi = psi.to_wigner_coherent(paths, seed=rng.randint(0, 2**32-1))

    bs = BeamSplitter(thr, psi.data, 0, 1, f_detuning=f_detuning, f_rabi=f_rabi)

    # Prepare integrator components
    drift = get_drift(state_dtype, grid, states, freqs, scattering, losses, wigner=True)
    diffusion = get_diffusion(state_dtype, grid, 2, losses)

    stepper = stepper_cls(
        grid.shape, grid.box, drift,
        kinetic_coeff=1j * const.HBAR / (2 * states[0].m),
        ensembles=paths,
        diffusion=diffusion)

    wiener = Wiener(stepper.parameter.dW, 1. / grid.dV, seed=rng.randint(0, 2**32-1))
    integrator = Integrator(thr, stepper, wiener=wiener, profile=True)

    # Integrate
    psi_temp = thr.empty_like(psi.data)
    psi_sampler = PsiSampler(thr, grid, psi_temp, bs, wigner=True)
    bs(psi.data, 0, numpy.pi / 2)
    result, info = integrator.fixed_step(
        psi.data, 0, interval, steps, samples=samples, samplers=[psi_sampler])

    N_mean = result['N_mean']
    N_exact = N * numpy.exp(-gamma * result['time'] * 2)

    return dict(
        errors=info.errors,
        N_diff=abs(N_mean.sum(1)[-1] - N_exact[-1]) / N_exact[-1],
        t_integration=info.timings.integration,
        )


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

    import pickle
    import os.path
    fname = 'convergence.pickle'

    if not os.path.exists(fname):
        with open(fname, 'wb') as f:
            pickle.dump({}, f, protocol=2)

    step_nums = [1, 2, 3, 5, 7]
    step_nums = sum(([1000 * num * 10 ** pwr for num in step_nums] for pwr in range(3)), [])

    for stepper_cls in steppers:
        for steps in step_nums:
            result = run_test(thr, stepper_cls, steps)
            print(result)

            with open(fname, 'rb') as f:
                results = pickle.load(f)
            label = stepper_cls.abbreviation
            if label not in results:
                results[label] = {}
            results[label][steps] = result
            with open(fname, 'wb') as f:
                pickle.dump(results, f, protocol=2)
