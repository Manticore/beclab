import numpy

from reikna.cluda import dtypes, functions
from reikna.core import Computation, Parameter, Annotation, Type
from reikna.algorithms import PureParallel


def get_splitter(state_arr, comp1, comp2):
    real_dtype = dtypes.real_for(state_arr.dtype)
    trajectories = state_arr.shape[0]
    return PureParallel(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('thetas', Annotation(Type(real_dtype, trajectories))),
            Parameter('phis', Annotation(Type(real_dtype, trajectories)))],
        """
        <%
            trajectory = idxs[0]
            coords = ", ".join(idxs[1:])
            r_ctype = dtypes.ctype(dtypes.real_for(output.dtype))
        %>
        %for comp in range(components):
        const ${output.ctype} psi_${comp} = ${input.load_idx}(${trajectory}, ${comp}, ${coords});
        %endfor

        const ${thetas.ctype} theta = ${thetas.load_idx}(${trajectory});
        const ${phis.ctype} phi = ${phis.load_idx}(${trajectory});

        ${r_ctype} sin_half_theta = sin(theta / 2);
        ${r_ctype} cos_half_theta = cos(theta / 2);

        ${output.ctype} minus_i = COMPLEX_CTR(${output.ctype})(0, -1);
        ${output.ctype} k2 = ${mul_sr}(
            ${mul_ss}(minus_i, ${polar_unit}(-phi)),
            sin_half_theta);
        ${output.ctype} k3 = ${mul_sr}(
            ${mul_ss}(minus_i, ${polar_unit}(phi)),
            sin_half_theta);

        const ${output.ctype} psi_${comp1}_new =
            ${mul_sr}(psi_${comp1}, cos_half_theta) + ${mul_ss}(psi_${comp2}, k2);
        const ${output.ctype} psi_${comp2}_new =
            ${mul_ss}(psi_${comp1}, k3) + ${mul_sr}(psi_${comp2}, cos_half_theta);

        %for comp in range(components):
        %if comp in (comp1, comp2):
        ${output.store_idx}(${trajectory}, ${comp}, ${coords}, psi_${comp}_new);
        %else:
        ${output.store_idx}(${trajectory}, ${comp}, ${coords}, psi_${comp});
        %endif
        %endfor
        """,
        guiding_array=(state_arr.shape[0],) + state_arr.shape[2:],
        render_kwds=dict(
            comp1=comp1,
            comp2=comp2,
            components=state_arr.shape[1],
            polar_unit=functions.polar_unit(real_dtype),
            mul_ss=functions.mul(state_arr.dtype, state_arr.dtype),
            mul_sr=functions.mul(state_arr.dtype, real_dtype)))


class BeamSplitterMatrix(Computation):

    def __init__(self, state_arr, comp1, comp2):

        real_dtype = dtypes.real_for(state_arr.dtype)
        trajectories = state_arr.shape[0]
        Computation.__init__(self,
            [Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('thetas', Annotation(Type(real_dtype, trajectories))),
            Parameter('phis', Annotation(Type(real_dtype, trajectories)))])

        self._splitter = get_splitter(state_arr, comp1, comp2)

    def _build_plan(self, plan_factory, device_params, output, input_, thetas, phis):

        plan = plan_factory()
        plan.computation_call(self._splitter, output, input_, thetas, phis)
        return plan


class BeamSplitter:
    r"""
    Beam splitter for a multi-component BEC.
    The effect of an oscillator is approximated by the application of the matrix

    .. math::

        \begin{pmatrix}
            \Psi^\prime_1 \\ \Psi^\prime_2
        \end{pmatrix} =
        \begin{pmatrix}
            \cos \frac{\theta}{2} & -i e^{-i \phi} \sin \frac{\theta}{2} \\
            -i e^{i \phi} \sin \frac{\theta}{2} & \cos \frac{\theta}{2}
        \end{pmatrix}
        \begin{pmatrix}
            \Psi_1 \\ \Psi_2
        \end{pmatrix},

    where :math:`\phi = \delta t + \alpha`.

    :param wfs_meta: a :py:class:`~beclab.wavefunction.WavefunctionSetMetadata` object.
    :param comp1_num: the number of the first affected component.
    :param comp2_num: the number of the second affected component.
    :param starting_phase: initial phase :math:`\alpha` of the oscillator (in radians).
    :param f_detuning: detuning frequency :math:`\delta` (in Hz).
    :param f_rabi: Rabi frequency :math:`\Omega` of the oscillator.
        Connected to the time of the pulse as :math:`\theta = \Omega t_{\mathrm{pulse}}`.
    :param seed: RNG seed for calls with ``theta_noise > 0``.
    """

    def __init__(self, wfs_meta, comp1_num=0, comp2_num=1,
            starting_phase=0, f_detuning=0, f_rabi=0, seed=None):

        self._rng = numpy.random.RandomState(seed=seed)
        self._starting_phase = starting_phase
        self._detuning = 2 * numpy.pi * f_detuning
        self._f_rabi = f_rabi
        self._trajectories = wfs_meta.trajectories
        self._thread = wfs_meta.thread
        self._real_dtype = dtypes.real_for(wfs_meta.dtype)
        self._splitter = BeamSplitterMatrix(
            wfs_meta.data, comp1_num, comp2_num).compile(wfs_meta.thread)

    def __call__(self, wfs_data, t, theta, theta_noise=0, phi_noise=0):
        r"""
        Applies beam splitter to an on-device wavefunction array.

        :param wfs_data: a Reikna ``Array`` (with the same dtype and shape as ``wfs_meta.data``).
        :param t: time of application (affects the total phase :math:`\phi`).
        :param theta: rotation angle :math:`\theta`, in radians.
        :param theta_noise: standard deviation of :math:`\theta` values for different trajectories.
        :param phi_noise: standard deviation of :math:`\phi` values for different trajectories.
        """
        phi = t * self._detuning + self._starting_phase
        t_pulse = (theta / numpy.pi / 2.0) / self._f_rabi

        # TODO: use GPU RNG?
        if phi_noise > 0.0:
            phis = self._rng.normal(size=self._trajectories, scale=phi_noise, loc=phi)
        else:
            phis = numpy.ones(self._trajectories) * phi

        if theta_noise > 0.0:
            thetas = self._rng.normal(size=self._trajectories, scale=theta_noise, loc=theta)
        else:
            thetas = numpy.ones(self._trajectories) * theta

        thetas = self._thread.to_device(thetas.astype(self._real_dtype))
        phis = self._thread.to_device(phis.astype(self._real_dtype))

        self._splitter(wfs_data, wfs_data, thetas, phis)
        return t + t_pulse
