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
    FixedStepIntegrator, Wiener, SSCDStepper, CDStepper, Drift, Diffusion)


def get_drift(complex_dtype, U, gamma, dx, wigner=False):
    return Drift(
        Module.create(
            """
            <%
                r_dtype = dtypes.real_for(c_dtype)
                c_ctype = dtypes.ctype(c_dtype)
                r_ctype = dtypes.ctype(r_dtype)
            %>
            INLINE WITHIN_KERNEL ${c_ctype} ${prefix}0(
                const int idx_x,
                const ${c_ctype} psi,
                ${r_ctype} t)
            {
                return ${mul_cc}(
                    COMPLEX_CTR(${c_ctype})(
                        -${gamma},
                        -(${U} * (${norm}(psi) - ${correction}))),
                    psi
                );
            }
            """,
            render_kwds=dict(
                c_dtype=complex_dtype,
                U=U,
                gamma=gamma,
                mul_cc=functions.mul(complex_dtype, complex_dtype),
                norm=functions.norm(complex_dtype),
                correction=1. / dx if wigner else 0
                )),
        complex_dtype, components=1)


def get_diffusion(complex_dtype, gamma):
    return Diffusion(
        Module.create(
            """
            <%
                r_dtype = dtypes.real_for(c_dtype)
                c_ctype = dtypes.ctype(c_dtype)
                r_ctype = dtypes.ctype(r_dtype)
            %>
            INLINE WITHIN_KERNEL ${c_ctype} ${prefix}0_0(
                const int idx_x,
                const ${c_ctype} psi,
                ${r_ctype} t)
            {
                return COMPLEX_CTR(${c_ctype})(${numpy.sqrt(gamma)}, 0);
            }
            """,
            render_kwds=dict(
                mul_cr=functions.mul(complex_dtype, dtypes.real_for(complex_dtype)),
                c_dtype=complex_dtype,
                gamma=gamma)),
        complex_dtype, components=1, noise_sources=1)


class PsiCollector:

    def __init__(self, dx, wigner=False):
        self._wigner = wigner
        self._dx = dx

    def __call__(self, psi, t):
        psi = psi.get()

        # (components, ensembles, x_points)
        density = numpy.abs(psi) ** 2 - (0.5 / self._dx if self._wigner else 0)

        N = density.sum(2) * self._dx
        N_mean = N.mean(1)[0]
        N_err = N.std(1)[0] / numpy.sqrt(N.shape[1])

        return dict(
            psi=psi,
            density=density.mean(1)[0],
            N_mean=N_mean, N_err=N_err
            )


def run_test(thr, label, stepper_cls, no_losses=False, wigner=False):

    print()
    print(
        "*** Running " + label + ", wigner=" + str(wigner) +
        ", no_losses=" + str(no_losses) + " test")
    print()

    # Simulation parameters
    lattice_size = 128 # spatial lattice points
    domain = (-7., 7.) # spatial domain
    paths = 128 # simulation paths
    interval = 0.5 # time interval
    samples = 100 # how many samples to take during simulation
    steps = samples * 100 # number of time steps (should be multiple of samples)
    gamma = 0.0 if no_losses else 0.2
    soliton_height = 10.0
    U = -2. / soliton_height ** 2
    complex_dtype = numpy.complex128
    float_dtype = dtypes.real_for(complex_dtype)
    seed = 1234

    # Lattice
    x = numpy.linspace(domain[0], domain[1], lattice_size, endpoint=False)
    dx = x[1] - x[0]

    # Initial state
    psi0 = soliton_height / numpy.cosh(x)
    N0 = (numpy.abs(psi0) ** 2).sum() * dx
    n_max = (numpy.abs(psi0) ** 2).max()
    print("N0 =", N0)

    # Initial noise
    if wigner:
        numpy.random.seed(seed)
        random_normals = (
            numpy.random.normal(size=(1, paths, lattice_size)) +
            1j * numpy.random.normal(size=(1, paths, lattice_size))) / 2
        fft_scale = numpy.sqrt(dx / lattice_size)
        psi0 = numpy.fft.ifft(random_normals) / fft_scale + psi0
    else:
        psi0 = numpy.tile(psi0, (1, 1, 1))

    psi_gpu = thr.to_device(psi0.astype(complex_dtype))

    # Prepare integrator components
    drift = get_drift(complex_dtype, U, gamma, dx, wigner=wigner)
    stepper = stepper_cls((lattice_size,), (domain[1] - domain[0],), drift,
        kinetic_coeff=1,
        ensembles=paths if wigner else 1,
        diffusion=get_diffusion(complex_dtype, gamma) if wigner else None)

    if wigner:
        wiener = Wiener(stepper.parameter.dW, 1. / dx, seed=seed)
    integrator = FixedStepIntegrator(
        thr, stepper,
        wiener=wiener if wigner else None)

    # Integrate
    psi_collector = PsiCollector(dx, wigner=wigner)
    result, errors = integrator(
        psi_gpu, 0, interval, steps, samples=samples, collectors=[psi_collector])

    N_mean = result['N_mean']
    N_err = result['N_err']
    density = result['density']
    N_exact = N0 * numpy.exp(-gamma * result['time'] * 2)

    suffix = ('_wigner' if wigner else '') + ('_no-losses' if no_losses else '') + '_' + label

    # Plot density
    fig = plt.figure()
    s = fig.add_subplot(111)
    s.imshow(density.T, interpolation='nearest', origin='lower', aspect='auto',
        extent=(0, interval) + domain, vmin=0, vmax=n_max)
    s.set_xlabel('$t$')
    s.set_ylabel('$x$')
    fig.savefig('soliton_density' + suffix + '.pdf')

    # Plot population
    fig = plt.figure()
    s = fig.add_subplot(111)
    s.plot(result['time'], N_mean, 'b-')
    if wigner:
        s.plot(result['time'], N_mean + N_err, 'b--')
        s.plot(result['time'], N_mean - N_err, 'b--')
    s.plot(result['time'], N_exact, 'k--')
    s.set_ylim(-N0 * 0.1, N0 * 1.1)
    s.set_xlabel('$t$')
    s.set_ylabel('$N$')
    fig.savefig('soliton_N' + suffix + '.pdf')

    # Compare with the analytical solution
    if not wigner and no_losses:
        psi = result['psi']
        tt = numpy.linspace(0, interval, samples + 1, endpoint=True).reshape(samples + 1, 1, 1, 1)
        xx = x.reshape(1, 1, 1, x.size)
        psi_exact = soliton_height / numpy.cosh(xx) * numpy.exp(1j * tt)
        diff = numpy.linalg.norm(psi - psi_exact) / numpy.linalg.norm(psi_exact)
        print("Difference with an exact solution:", diff)
        assert diff < 1e-3

    # Check the population decay
    sigma = numpy.linalg.norm(N_err) if wigner else N_mean * 1e-6
    max_diff = (numpy.abs(N_mean - N_exact) / sigma).max()
    print("Maximum difference with an exact population decay:", max_diff, "sigma")
    assert max_diff < 1


if __name__ == '__main__':

    # Run integration
    api = cluda.ocl_api()
    thr = api.Thread.create()

    tests = [
        ("SS-CD", SSCDStepper),
        ("CD", CDStepper)
    ]

    for label, cls in tests:
        run_test(thr, label, cls, no_losses=True, wigner=False)
        run_test(thr, label, cls, wigner=False)
        run_test(thr, label, cls, wigner=True)
