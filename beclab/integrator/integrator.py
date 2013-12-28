from __future__ import print_function, division

import numpy
import sys
import time

from progressbar import Bar, ETA, Timer, Percentage, ProgressBar, Widget, WidgetHFill


_range = xrange if sys.version_info[0] < 3 else range


class StatefulLabel(Widget):

    def __init__(self, display):
        keys = []
        formatters = []
        for key in display:
            if isinstance(key, tuple):
                key, formatter = key
                formatter = "{" + key + ":" + formatter + "}"
            else:
                formatter = "{" + key + "}"
            keys.append(key)
            formatters.append(formatter)

        format_str = ", ".join(key + ": " + formatter for key, formatter in zip(keys, formatters))

        self.keys = keys
        self.format_str = format_str
        self.values = None

    def set(self, sample_dict):
        if self.values is None:
            self.values = {}
        for key in self.keys:
            self.values[key] = sample_dict[key]

    def update(self, pbar):
        if self.values is not None:
            return self.format_str.format(**self.values)
        else:
            return "(not initialized)"


class HFill(WidgetHFill):

    def update(self, hbar, width):
        return ' ' * width


class StopIntegration(Exception):
    pass


class IntegrationError(Exception):
    pass


def _transpose_results(results):
    new_results = {key:[] for key in results[0].keys()}
    for res in results:
        for key in res:
            new_results[key].append(res[key])

    return {key:numpy.array(val) for key, val in new_results.items()}


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

    def __init__(self, timings, errors):
        self.errors = errors
        self.timings = timings


class Integrator:

    def __init__(self, thr, stepper, wiener=None, verbose=True, profile=False):

        self.thr = thr
        self.verbose = verbose
        self.profile = profile

        self.stepper = stepper.compile(thr)
        # TODO: temporary dW array can be avoided if wiener-stepper is called in a computation
        if wiener is not None:
            self.noise = True
            self._w = wiener
            self.wiener = wiener.compile(thr)
            self.wiener_double = wiener.double_step().compile(thr)
            self.dW = self.thr.empty_like(wiener.parameter.dW)
            self.dW_state = self.thr.empty_like(wiener.parameter.state)
        else:
            self.noise = False

    def _sample(self, data, t, samplers):
        if self.profile:
            self.thr.synchronize()
        t1 = time.time()
        sample_dict = dict(time=t)
        stop_integration = False
        for sampler in samplers:
            try:
                sample = sampler(data, t)
            except StopIntegration as e:
                sample = e.args[0]
                stop_integration = True
            sample_dict.update(sample)
        t_samplers = time.time() - t1
        return sample_dict, stop_integration, t_samplers

    def _integrate(self, data_out, data_in, double_step, t_start, dt, steps, samples=None,
            samplers=None, verbose=False, filters=None):

        stepper = self.stepper
        noise_dt = dt / 2 if double_step else dt

        results = []
        step = 0
        sample = 0
        t = t_start

        t_samplers = 0
        t_integration_start = time.time()

        if self.noise:
            wiener = self.wiener_double if double_step else self.wiener
            dW_state = self.dW_state
            dW = self.dW
            self.thr.to_device(
                numpy.zeros(wiener.parameter.state.shape, wiener.parameter.state.dtype),
                dest=dW_state)

        if verbose:
            title = 'Double step ' if double_step else 'Normal step '
            val_step = steps // 100 if steps >= 100 else 1
            pbar = ProgressBar(
                widgets=[title, Percentage(), Bar(), ETA()], maxval=steps - 1).start()

        for step in _range(steps):
            if step == 0:
                _data_in = data_in
            else:
                _data_in = data_out

            if self.noise:
                wiener(dW_state, dW, noise_dt)
                stepper(data_out, _data_in, dW, t, dt)
            else:
                stepper(data_out, _data_in, t, dt)

            # Calling every second step for normal step so that filters are executed
            # exactly at the same places, and the convergence can be measured.
            if double_step or step % 2 == 0:
                for filter_ in filters:
                    filter_(data_out, t)

            t += dt

            if verbose and steps % val_step == 0:
                pbar.update(step)

            if samples is not None:
                if (step + 1) % (steps // samples) == 0:
                    sample_dict, stop_integration, t_sampler = self._sample(data_out, t, samplers)
                    results.append(sample_dict)
                    t_samplers += t_sampler
                    if stop_integration:
                        break

        if verbose:
            pbar.finish()

        t_total = time.time() - t_integration_start

        if verbose and not double_step:
            print("Samplers time: {f:.1f}%".format(f=t_samplers / t_total * 100))

        return results, t_total - t_samplers, t_samplers

    def fixed_step(self, data_dev, t_start, t_end, steps, samples=1, samplers=None, filters=None):

        if samplers is None:
            samplers = []
        if filters is None:
            filters = []

        assert steps % samples == 0
        assert steps % 2 == 0
        dt = (t_end - t_start) / steps

        if self.verbose:
            print("Integrating from " + str(t_start) + " to " + str(t_end))

        # double step (to estimate the convergence)
        data_double_dev = self.thr.copy_array(data_dev)
        results_double, t_double, t_samplers_double = self._integrate(
            data_double_dev, data_double_dev, True, t_start, dt * 2, steps // 2,
            samplers=samplers, samples=1, verbose=self.verbose, filters=filters)

        # actual integration
        sample_start, _, t_samplers_start = self._sample(data_dev, t_start, samplers)
        results, t_normal, t_samplers_normal = self._integrate(
            data_dev, data_dev, False, t_start, dt, steps,
            samples=samples, samplers=samplers, verbose=self.verbose, filters=filters)
        results = [sample_start] + results

        # FIXME: need to cache Norm computations and perform this on device
        # calculate result errors
        errors = {}
        for key in results[-1]:
            if key == 'time':
                continue
            res_double = results_double[-1][key]
            res = results[-1][key]
            error_norm = numpy.linalg.norm(res)
            if error_norm > 0:
                error = numpy.linalg.norm(res_double - res) / error_norm
            else:
                error = 0
            errors[key] = error
            if self.verbose:
                print("Error in " + key + ":", errors[key])

        timings = Timings(
            normal=t_normal, double=t_double,
            samplers=t_samplers_normal + t_samplers_double + t_samplers_start)
        info = IntegrationInfo(timings, errors)

        return _transpose_results(results), info

    def adaptive_step(
            self, data_dev, t_start, t_sample,
            starting_steps=2,
            t_end=None, samplers=None, convergence=None, filters=None,
            display=None,
            steps_limit=10000):

        if self.noise:
            # TODO: in order to support it we must somehow implement noise splitting
            # (so that the convergence improves steadily when we decrease the time step).
            # See Wilkie & Cetinbas, 2004 (doi:10.1016/j.physleta.2005.01.064).
            raise NotImplementedError("Adaptive step for stochastic equations is not supported")

        assert starting_steps % 2 == 0

        if samplers is None:
            samplers = []
        if filters is None:
            filters = []
        if convergence is None:
            convergence = {}

        if self.verbose:
            if display is not None:
                label = StatefulLabel(display)
            else:
                label = StatefulLabel([('time', '.3f')])

        if self.verbose:
            print(
                "Integrating from " + str(t_start) + " to " +
                (str(t_end) if t_end is not None else "infinity"))

        data_try = self.thr.empty_like(data_dev)
        data_double_try = self.thr.empty_like(data_dev)

        t = t_start
        steps = starting_steps

        sample_start, _, t_samplers = self._sample(data_dev, t, samplers)
        results = [sample_start]
        timings = Timings(samplers=t_samplers)
        if self.verbose:
            label.set(sample_start)

        if self.verbose:
            if t_end is None:
                widgets = [label, ' ', HFill(), Timer()]
                maxval = None
            else:
                widgets = [label, ' ', HFill(), Percentage(), ' ', ETA()]
                maxval = t_end - t_start

            pbar = ProgressBar(widgets=widgets, maxval=maxval).start()

        while True:
            dt = t_sample / steps
            _, t_double, _ = self._integrate(
                data_double_try, data_dev, True, t, dt * 2, steps // 2, verbose=False,
                filters=filters)
            _, t_normal, _ = self._integrate(
                data_try, data_dev, False, t, dt, steps, verbose=False,
                filters=filters)

            t += t_sample

            sample_normal, stop_integration, t_samplers_normal = self._sample(data_try, t, samplers)
            sample_double, _, t_samplers_double = self._sample(data_double_try, t, samplers)

            timings += Timings(
                normal=t_normal, double=t_double,
                samplers=t_samplers_normal + t_samplers_double)

            converged = True
            for key in convergence:
                res = sample_normal[key]
                res_double = sample_double[key]
                error = numpy.linalg.norm(res_double - res) / numpy.linalg.norm(res)
                if error > convergence[key]:
                    converged = False
                    break

            if converged:
                self.thr.copy_array(data_try, dest=data_dev)

                results.append(sample_normal)
                if self.verbose:
                    label.set(sample_normal)
                    if t_end is None:
                        pbar.update(t - t_start)
                    else:
                        pbar.update(min(maxval, t - t_start))

                if stop_integration or (t_end is not None and t >= t_end):
                    break

                if steps > 2:
                    steps //= 2
            else:
                t -= t_sample
                steps *= 2
                if steps > steps_limit:
                    raise IntegrationError("Number of steps per sample is greater than the limit")

        if self.verbose:
            pbar.finish()

        # calculate result errors
        errors = {}
        for key in sorted(sample_normal):
            if key == 'time':
                continue
            res_double = sample_double[key]
            res = sample_normal[key]
            error_norm = numpy.linalg.norm(res)
            if error_norm > 0:
                error = numpy.linalg.norm(res_double - res) / error_norm
            else:
                error = 0
            errors[key] = error
            if self.verbose:
                print("Error in " + key + ":", errors[key] * (t - t_start) / t_sample)

        info = IntegrationInfo(timings, errors)

        return _transpose_results(results), info
