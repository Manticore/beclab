import sys
import numpy


_range = xrange if sys.version_info[0] < 3 else range


class StopIntegration(Exception):
    pass


class IntegrationError(Exception):
    pass


class Timings:

    def __init__(self, normal=0, double=0, samplers=0):
        self.integration = normal + double
        self.samplers = samplers
        self.normal = normal
        self.double = double

    def __add__(self, other):
        return Timings(
            normal=self.normal + other.normal,
            double=self.double + other.double,
            samplers=self.samplers + other.samplers)


class IntegrationInfo:

    def __init__(self, timings, strong_errors, weak_errors, steps):
        self.weak_errors = weak_errors
        self.strong_errors = strong_errors
        self.timings = timings
        self.steps = steps


class Sampler:

    def __init__(self, no_mean=False, no_stderr=False, no_values=False):
        self.no_mean = no_mean
        self.no_stderr = no_stderr
        self.no_values = no_values


def sample(data, t, samplers):
    sample_dict = {}
    stop_integration = False

    for key, sampler in samplers.items():

        try:
            sample = sampler(data, t)
        except StopIntegration as e:
            sample = e.args[0]
            stop_integration = True

        sample_dict[key] = dict(trajectories=sample.shape[0], time=t)
        if not sampler.no_values:
            sample_dict[key]['values'] = sample
        if not sampler.no_mean:
            sample_dict[key]['mean'] = sample.mean(0)
        if not sampler.no_stderr:
            sample_dict[key]['stderr'] = sample.std(0) / numpy.sqrt(sample.shape[0])

    t_samplers = time.time() - t1
    return sample_dict, stop_integration, t_samplers


def calculate_errors(sample_normal, sample_double, strong_keys, weak_keys):

    # FIXME: performance can be improved by calculating norms on GPU

    weak_errors = {}
    strong_errors = {}

    for key in weak_keys:
        mean_normal = sample_normal[key]['mean']
        mean_double = sample_double[key]['mean']
        error_norm = numpy.linalg.norm(mean_normal)
        if error_norm > 0:
            error = numpy.linalg.norm(mean_normal - mean_double) / error_norm
        else:
            error = 0
        weak_errors[key] = error

    for key in strong_keys:
        values_normal = sample_normal[key]['values']
        values_double = sample_double[key]['values']

        errors = []
        for i in _range(values_normal.shape[0]):
            value_normal = values_normal[i]
            value_double = values_double[i]

            error_norm = numpy.linalg.norm(value_normal)
            if error_norm > 0:
                error = numpy.linalg.norm(value_normal - value_double) / error_norm
            else:
                error = 0
            errors.append(error)

        strong_errors[key] = max(errors)

    return strong_errors, weak_errors


def transpose_results(results):
    new_results = {}
    for key in results[0]:
        new_results[key] = dict(trajectories=results[0][key]['trajectories'])
        for val_key in results[0][key]:
            if val_key != 'trajectories':
                new_results[key][val_key] = []

    for res in results:
        for key in res:
            for val_key in res[key]:
                if val_key != 'trajectories':
                    new_results[key][val_key].append(res[key][val_key])

    for key in new_results:
        for val_key in new_results[key]:
            if val_key != 'trajectories':
                new_results[key][val_key] = numpy.array(new_results[key][val_key])

    return new_results


def join_results(results1, results2):
    assert set(results1.keys()) == set(results2.keys())
    full_results = {}
    for key in results1:
        r1 = results1[key]
        r2 = results2[key]
        tr1 = r1['trajectories']
        tr2 = r2['trajectories']

        assert all(r1['time'] == r2['time'])
        full_results[key] = dict(time=r1['time'], trajectories=tr1 + tr2)

        if 'values' in r1:
            full_results[key]['values'] = numpy.concatenate([r1['values'], r2['values']], axis=1)
        if 'mean' in r1:
            mean1 = r1['mean']
            mean2 = r2['mean']
            full_results[key]['mean'] = (mean1 * tr1 + mean2 * tr2) / (tr1 + tr2)
        if 'stderr' in r1:
            err1 = r1['stderr']
            err2 = r2['stderr']
            full_results[key]['stderr'] = numpy.sqrt(
                (err1**2 * tr1**2 + err2**2 * tr2**2)) / (tr1 + tr2)

    return full_results

