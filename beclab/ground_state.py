import numpy

from reikna.cluda import dtypes
import reikna.cluda.functions as functions
from reikna.algorithms import PureParallel
from reikna.core import Parameter, Annotation, Transformation


import beclab.constants as const
from beclab.wavefunction import WavefunctionSet
from beclab.integrator import Sampler, RK46NLStepper, Integrator
from beclab.modules import get_drift
from beclab.meters import PopulationMeter, EnergyMeter


def tf_ground_state(thr, grid, dtype, system, Ns):

    if grid.dimensions != 3:
        raise NotImplementedError()

    wfs = WavefunctionSet(thr, grid, dtype, components=len(system.components))
    psi_TF = numpy.empty(wfs.shape, wfs.dtype)
    for i, comp in enumerate(system.components):
        N = Ns[i]

        if N == 0:
            psi_TF[i] = 0
            continue

        mu = const.mu_tf_3d(system.potential.trap_frequencies, N, comp.m, system.scattering[i, i])

        V = system.get_potential(grid, i)
        psi_TF[i] = numpy.sqrt((mu - V).clip(0) / system.scattering[i, i])

        # renormalize to account for coarse grids
        N0 = (numpy.abs(psi_TF[i]) ** 2).sum() * grid.dV
        psi_TF[i] *= numpy.sqrt(N / N0)

    wfs.fill_with(psi_TF)
    return wfs


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


def it_ground_state(thr, grid, dtype, system, Ns, E_diff=1e-9, E_conv=1e-9, sample_time=1e-3,
        samplers=None, return_info=False, stepper_cls=RK46NLStepper):

    if grid.dimensions != 3:
        raise NotImplementedError()

    # Initial TF state
    psi = tf_ground_state(thr, grid, dtype, system, Ns)

    drift = get_drift(dtype, grid, system.components, system.potential.trap_frequencies,
        system.scattering, imaginary_time=True)

    stepper = stepper_cls(
        grid.shape, grid.box, drift,
        kinetic_coeff=(-1j * system.kinetic_coeff).real)
    integrator = Integrator(thr, stepper, verbose=True)

    # Integrate
    energy_meter = EnergyMeter(thr, psi, system)
    population_meter = PopulationMeter(thr, psi)
    e_sampler = EnergySampler(energy_meter)
    e_stopper = PropagationStop(energy_meter, limit=E_diff)
    prop_samplers = dict(E_conv=e_sampler, E=e_stopper)
    if samplers is not None:
        prop_samplers.update(samplers)

    psi_filter = PsiFilter(thr, psi, Ns, population_meter)

    result, info = integrator.adaptive_step(
        psi.data, 0, sample_time,
        display=['time', 'E'],
        samplers=prop_samplers,
        filters=[psi_filter],
        convergence=dict(E_conv=E_conv))

    del results['E_conv']

    if return_info:
        return psi, results, info
    else:
        return psi
