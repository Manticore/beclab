import numpy

import reikna.cluda.dtypes as dtypes


def get_ksquared(shape, box):
    ks = [
        2 * numpy.pi * numpy.fft.fftfreq(size, length / size)
        for size, length in zip(shape, box)]

    if len(shape) > 1:
        full_ks = numpy.meshgrid(*ks, indexing='ij')
    else:
        full_ks = ks

    return sum([full_k ** 2 for full_k in full_ks])


def get_kprop_trf(state_arr, kprop_arr, kinetic_coeff):
    kcoeff_dtype = dtypes.detect_type(kinetic_coeff)
    return Transformation(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('kprop', Annotation(kprop_arr, 'i')),
            Parameter('dt', Annotation(kprop_arr.dtype))],
        """
        ${kprop.ctype} kprop = ${kprop.load_idx}(${', '.join(idxs[2:])});
        ${output.store_same}(${mul}(${input.load_same}, ${kinetic_coeff}, kprop * ${dt}));
        """,
        render_kwds=dict(
            kinetic_coeff=dtypes.c_constant(kinetic_coeff, kcoeff_dtype),
            mul=functions.mul(
                state_arr.dtype, kcoeff_dtype, kprop_arr.dtype, out_dtype=state_arr.dtype)))


def get_kprop_exp_trf(state_arr, kprop_arr, kinetic_coeff):
    kcoeff_dtype = dtypes.detect_type(kinetic_coeff)
    return Transformation(
        [
            Parameter('output', Annotation(state_arr, 'o')),
            Parameter('input', Annotation(state_arr, 'i')),
            Parameter('kprop', Annotation(kprop_arr, 'i')),
            Parameter('dt', Annotation(kprop_arr.dtype))],
        """
        ${kprop.ctype} kprop = ${kprop.load_idx}(${', '.join(idxs[2:])});
        ${output.ctype} kprop_exp = ${exp}(${mul_k}(kprop * ${dt}, ${kinetic_coeff}));
        ${output.store_same}(${mul}(${input.load_same}, kprop_exp));
        """,
        render_kwds=dict(
            kinetic_coeff=dtypes.c_constant(kinetic_coeff, kcoeff_dtype),
            mul_k=functions.mul(kprop_arr.dtype, kinetic_coeff.dtype, out_dtype=state_arr.dtype),
            exp=functions.exp(state_arr.dtype),
            mul=functions.mul(state_arr.dtype, state_arr.dtype)))
