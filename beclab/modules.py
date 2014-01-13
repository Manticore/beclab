import numpy

import reikna.cluda.dtypes as dtypes
import reikna.cluda.functions as functions
from reikna.cluda import Module

from beclab.integrator import Drift, Diffusion
import beclab.constants as const


def get_drift(state_dtype, dimensions, components, interactions=None, corrections=None,
        potential=None, losses=None, unitary_coefficient=1):
    """
    interactions, array(comps, comps): two-body elastic interaction constants.
    losses, [(kappa, [l_1, ..., l_comps])]: inelastic interaction constants
        (l_i specifies the number of atoms of component i participating in the interaction).
        Coefficients are the "theorist's" ones.
    """
    if losses is None:
        losses = []
    real_dtype = dtypes.real_for(state_dtype)

    if interactions is None:
        interactions = numpy.zeros((components, components))
    if corrections is None:
        corrections = numpy.zeros_like(interactions)

    return Drift(
        Module.create(
            """
            <%
                s_ctype = dtypes.ctype(s_dtype)
                r_ctype = dtypes.ctype(r_dtype)
                r_const = lambda x: dtypes.c_constant(x, r_dtype)
            %>
            %for comp in range(components):
            INLINE WITHIN_KERNEL ${s_ctype} ${prefix}${comp}(
                %for dim in range(dimensions):
                const int idx_${dim},
                %endfor
                %for c in range(components):
                const ${s_ctype} psi_${c},
                %endfor
                ${r_ctype} t)
            {
                // Potential
                %if potential is not None:
                const ${r_ctype} V = ${potential}${comp}(
                    %for dim in range(dimensions):
                    idx_${dim},
                    %endfor
                    t
                    );
                %else:
                const ${r_ctype} V = 0;
                %endif

                // Elastic interactions
                const ${r_ctype} U =
                    %for other_comp in range(components):
                    <%
                        g = interactions[comp, other_comp]
                        correction = corrections[comp, other_comp]
                    %>
                    %if g != 0:
                    + ${r_const(g)}
                        * (${norm}(psi_${other_comp}) + (${r_const(correction)}))
                    %endif
                    %endfor
                    ;

                // Losses
                <%
                    # Walk through the losses and collect all components
                    # for which we will need norm(psi_j).
                    norms = set()
                    for kappa, ls in losses:
                        if kappa > 0 and ls[comp] > 0:
                            if ls[comp] > 1:
                                norms.add(comp)
                            for other_comp in range(components):
                                if other_comp != comp and ls[other_comp] > 0:
                                    norms.add(other_comp)
                %>
                %for norm_comp in sorted(norms):
                const ${r_ctype} n_${norm_comp} = ${norm}(psi_${norm_comp});
                %endfor

                const ${s_ctype} L =
                    ${dtypes.c_constant(0, s_dtype)}
                    %for kappa, ls in losses:
                    %if kappa > 0 and ls[comp] > 0:
                    - ${mul_sr}(
                        psi_${comp},
                        ${r_const(kappa * ls[comp])}
                            %for other_comp in range(components):
                                <%
                                    pwr = ls[other_comp]
                                    if comp == other_comp:
                                        pwr -= 1
                                %>
                                %for i in range(pwr):
                                * n_${other_comp}
                                %endfor
                            %endfor
                        )
                    %endif
                    %endfor
                    ;

                const ${s_ctype} unitary = ${mul_ss}(
                    COMPLEX_CTR(${s_ctype})(
                        ${unitary_coefficient.real},
                        ${unitary_coefficient.imag}
                        ),
                    ${mul_sr}(psi_${comp}, V + U));

                return unitary + L;
            }
            %endfor
            """,
            render_kwds=dict(
                unitary_coefficient=unitary_coefficient,
                dimensions=dimensions,
                components=components,
                potential=potential,
                s_dtype=state_dtype,
                r_dtype=real_dtype,
                interactions=interactions,
                corrections=corrections,
                losses=losses,
                mul_ss=functions.mul(state_dtype, state_dtype),
                mul_sr=functions.mul(state_dtype, real_dtype),
                norm=functions.norm(state_dtype),
                )),
        state_dtype, components=interactions.shape[0])


def get_diffusion(state_dtype, dimensions, components, losses=None):

    # Preparing multi-argument multiplications here, since they are modules
    # and cannot be instantiated inside a template.
    real_dtype = dtypes.real_for(state_dtype)
    muls = []

    if losses is None:
        losses = []

    for _, ls in losses:
        if sum(ls) > 1:
            muls.append(
                functions.mul(
                    real_dtype, *([state_dtype] * (sum(ls) - 1)), out_dtype=state_dtype))
        else:
            muls.append(None)

    return Diffusion(
        Module.create(
            """
            <%
                r_dtype = dtypes.real_for(s_dtype)
                s_ctype = dtypes.ctype(s_dtype)
                r_ctype = dtypes.ctype(r_dtype)
                r_const = lambda x: dtypes.c_constant(x, r_dtype)
            %>
            %for comp in range(components):
            %for noise_source in range(noise_sources):
            <%
                kappa, ls = losses[noise_source]
                mul = muls[noise_source]
                coeff = numpy.sqrt(kappa)
            %>
            INLINE WITHIN_KERNEL ${s_ctype} ${prefix}${comp}_${noise_source}(
                %for dim in range(dimensions):
                const int idx_${dim},
                %endfor
                %for c in range(components):
                const ${s_ctype} psi_${c},
                %endfor
                ${r_ctype} t)
            {
                %if ls[comp] == 0:
                return COMPLEX_CTR(${s_ctype})(0, 0);
                %else:
                    %if sum(ls) > 1:
                    return ${conj}(${mul}(
                        ${r_const(coeff)}
                        %for other_comp in range(components):
                            <%
                                pwr = ls[other_comp]
                                if comp == other_comp:
                                    pwr -= 1
                            %>
                            %for i in range(pwr):
                            , psi_${other_comp}
                            %endfor
                        %endfor
                        ));
                    %else:
                    return COMPLEX_CTR(${s_ctype})(${r_const(coeff)}, 0);
                    %endif
                %endif
            }
            %endfor
            %endfor
            """,
            render_kwds=dict(
                s_dtype=state_dtype,
                losses=losses,
                dimensions=dimensions,
                components=components,
                noise_sources=len(losses),
                muls=muls,
                conj=functions.conj(state_dtype))),
        state_dtype, components=components, noise_sources=len(losses))
