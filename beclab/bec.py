import numpy

from reikna.cluda import Module

import beclab.constants as const
import beclab.integrator as integrator
from beclab.integrator import RK46NLStepper
from beclab.modules import get_drift, get_diffusion
from beclab.wavefunction import WavefunctionSet, WavefunctionSetMetadata
from beclab.samplers import EnergySampler, StoppingEnergySampler
from beclab.filters import NormalizationFilter


class HarmonicPotential:

    def __init__(self, dtype, freqs, displacements=None):
        if not isinstance(freqs, tuple):
            raise NotImplementedError

        self.trap_frequencies = freqs
        self.displacements = displacements

    def _get_coeffs(self, grid, components):
        return numpy.array([
            [
                components[comp_num].m * (2 * numpy.pi * freq) ** 2 / 2
                for freq in self.trap_frequencies]
            for comp_num in range(len(components))])

    def get_module(self, dtype, grid, components):

        if self.displacements is None:
            displacements = []
        else:
            displacements = self.displacements

        return Module.create(
            """
            <%
                r_dtype = dtypes.real_for(s_dtype)
                r_ctype = dtypes.ctype(r_dtype)
                r_const = lambda x: dtypes.c_constant(x, r_dtype)
            %>
            %for comp_num in range(coeffs.shape[0]):
            INLINE WITHIN_KERNEL ${r_ctype} ${prefix}${comp_num}(
                %for dim in range(grid.dimensions):
                const int idx_${dim},
                %endfor
                ${r_ctype} t)
            {
                %for dim in range(grid.dimensions):
                ${r_ctype} x_${dim} =
                    ${r_const(grid.xs[dim][0])} + ${r_const(grid.dxs[dim])} * idx_${dim};
                %endfor

                %for dcomp, ddim, value in displacements:
                %if comp_num == dcomp:
                x_${ddim} += ${r_const(value)};
                %endif
                %endfor

                return
                    %for dim in range(grid.dimensions):
                    + ${r_const(coeffs[comp_num, dim])} * x_${dim} * x_${dim}
                    %endfor
                    ;
            }
            %endfor
            """,
            render_kwds=dict(
                grid=grid,
                coeffs=self._get_coeffs(grid, components),
                displacements=displacements,
                s_dtype=dtype,
                ))

    def get_array(self, grid, components):
        coeffs = self._get_coeffs(grid, components)

        result = numpy.empty((len(components),) + grid.shape)
        for comp_num in range(len(components)):

            xxs = numpy.meshgrid(*grid.xs, indexing="ij")

            if self.displacements is not None:
                for dcomp, ddim, value in self.displacements:
                    if dcomp == comp_num:
                        xxs[ddim] += value

            result[comp_num] = sum(
                coeffs[comp_num, dim] * xxs[dim] ** 2 for dim in range(grid.dimensions))

        return result


class System:

    def __init__(self, components, interactions, losses=None, potential=None):

        self.potential = potential
        self.components = components
        self.interactions = interactions
        self.losses = losses

        self.kinetic_coeff = -const.HBAR ** 2 / (2 * components[0].m)


def box_for_tf(system, comp_num, N, border=1.2):
    m = system.components[comp_num].m
    g = system.interactions[comp_num, comp_num]
    mu = const.mu_tf_3d(system.potential.trap_frequencies, N, m, g)
    diameter = lambda f: (
        2.0 * border * numpy.sqrt(2.0 * mu / (m * (2 * numpy.pi * f) ** 2)))
    return tuple(diameter(f) for f in system.potential.trap_frequencies)


class ThomasFermiGroundState:

    def __init__(self, thr, dtype, grid, system):

        if grid.dimensions != 3:
            raise NotImplementedError()

        self.thr = thr
        self.dtype = dtype
        self.grid = grid
        self.system = system

        self.wfs_meta = WavefunctionSetMetadata(
            thr, dtype, grid, components=len(self.system.components))

    def __call__(self, Ns):

        assert len(Ns) == len(self.system.components)

        wfs = WavefunctionSet.for_meta(self.wfs_meta)
        psi_TF = numpy.empty(wfs.shape, wfs.dtype)

        V = self.system.potential.get_array(self.grid, self.system.components)

        for i, component in enumerate(self.system.components):
            N = Ns[i]

            if N == 0:
                psi_TF[i] = 0
                continue

            mu = const.mu_tf_3d(
                self.system.potential.trap_frequencies, N,
                component.m, self.system.interactions[i, i])

            psi_TF[i] = numpy.sqrt((mu - V[i]).clip(0) / self.system.interactions[i, i])

            # renormalize to account for coarse grids
            N0 = (numpy.abs(psi_TF[i]) ** 2).sum() * self.grid.dV
            psi_TF[i] *= numpy.sqrt(N / N0)

        wfs.fill_with(psi_TF)
        return wfs


class ImaginaryTimeGroundState:

    def __init__(self, thr, dtype, grid, system, stepper_cls=RK46NLStepper, verbose=True):

        if grid.dimensions != 3:
            raise NotImplementedError()

        self.thr = thr
        self.dtype = dtype
        self.grid = grid
        self.system = system

        self.tf_gen = ThomasFermiGroundState(thr, dtype, grid, system)
        self.wfs_meta = self.tf_gen.wfs_meta

        drift = get_drift(
            dtype, grid.dimensions, len(system.components),
            interactions=system.interactions,
            potential=system.potential.get_module(dtype, grid, system.components),
            unitary_coefficient=-1 / const.HBAR)

        stepper = stepper_cls(
            grid.shape, grid.box, drift,
            kinetic_coeff=-1 / const.HBAR * system.kinetic_coeff)

        self.integrator = integrator.Integrator(thr, stepper, verbose=verbose)

    def __call__(self, Ns, E_diff=1e-9, E_conv=1e-9, sample_time=1e-3,
            samplers=None, return_info=False):

        # Initial TF state
        psi = self.tf_gen(Ns)

        e_sampler = EnergySampler(psi, self.system)
        e_stopper = StoppingEnergySampler(psi, self.system, limit=E_diff)
        prop_samplers = dict(E_conv=e_sampler, E=e_stopper)
        if samplers is not None:
            prop_samplers.update(samplers)

        psi_filter = NormalizationFilter(psi, Ns)

        result, info = self.integrator.adaptive_step(
            psi.data, 0, sample_time,
            display=['time', 'E'],
            samplers=prop_samplers,
            filters=[psi_filter],
            convergence=dict(E_conv=E_conv))

        del result['E_conv']

        if return_info:
            return psi, result, info
        else:
            return psi


class Integrator:

    def __init__(self, thr, dtype, system,
            wigner=False, seed=None, stepper_cls=RK46NLStepper, trajectories=1):

        drift = get_drift(
            dtype, system.grid, system.components,
            system.scattering, system.losses, wigner=wigner, potential=system.potential.module)
        if wigner:
            diffusion = get_diffusion(
                dtype, system.grid, len(system.components), system.losses)
        else:
            diffusion = None

        stepper = stepper_cls(
            system.grid.shape, system.grid.box, drift,
            kinetic_coeff=system.kinetic_coeff,
            trajectories=trajectories,
            diffusion=diffusion)

        if wigner:
            wiener = Wiener(stepper.parameter.dW, 1. / system.grid.dV, seed=seed)

        self._integrator = Integrator(
            thr, stepper,
            wiener=wiener if wigner else None)

    def fixed_step(self, wfs, *args, **kwds):
        return self._integrator.fixed_step(wfs.data, *args, **kwds)

    def adaptive_step(self, wfs, *args, **kwds):
        return self._integrator.adaptive_step(wfs.data, *args, **kwds)
