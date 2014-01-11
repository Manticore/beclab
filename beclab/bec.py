import numpy

import beclab.constants as const


def get_potential_module(state_dtype, grid, coeffs, displacements=None):

    if displacements is None:
        displacements = []

    return Module.create(
            """
            <%
                r_dtype = dtypes.real_for(s_dtype)
                r_ctype = dtypes.ctype(r_dtype)
                r_const = lambda x: dtypes.c_constant(x, r_dtype)
            %>
            %for comp_num in range(coeffs.shape[0]):
            INLINE WITHIN_KERNEL ${s_ctype} ${prefix}${comp_num}(
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
                coeffs=coeffs,
                displacements=displacements,
                s_dtype=state_dtype,
                ))


class HarmonicPotential:

    def __init__(self, dtype, grid, components, freqs, displacements=None):
        if not isinstance(freqs, tuple):
            raise NotImplementedError

        self._grid = grid
        self.trap_frequencies = freqs
        self._displacements = displacements

        self._coeffs = numpy.array([
            [components[comp_num].m * (2 * numpy.pi * freq) ** 2 / 2 for freq in freqs]
            for comp_num in range(len(components))])

        self.module = get_potential_module(dtype, grid, coeffs, displacements=displacements)

    def get_array(self, comp_num):
        xxs = numpy.meshgrid(*grid.xs, indexing="ij")
        for dcomp, ddim, value in displacements:
            if dcomp == comp_num:
                xxs[ddim] += value

        return sum(coeffs[comp_num, dim] * xxs[dim] ** 2 for dim in range(grid.dimensions))


class System:

    def __init__(self, components, scattering, potential=None):

        self.potential = potential
        self.components = components
        self.scattering = scattering

        self.kinetic_coeff = 1j * const.HBAR / (2 * components[0].m)


def box_for_tf(system, comp_num, N, border=1.2):
    component = system.components[comp_num]
    freqs = system.potential.trap_frequencies
    mu = const.mu_tf_3d(freqs, N, component.m, system.scattering[comp_num, comp_num])
    diameter = lambda f: 2.0 * border * numpy.sqrt(2.0 * mu / (comp.m * (2 * numpy.pi * f) ** 2))
    return tuple(diameter(f) for f in freqs)
