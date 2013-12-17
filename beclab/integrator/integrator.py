from __future__ import print_function, division

import numpy
import sys
import time

from progressbar import Bar, ETA, Percentage, ProgressBar


_range = xrange if sys.version_info[0] < 3 else range


def _transpose_results(results):
    new_results = {key:[] for key in results[0].keys()}
    for res in results:
        for key in res:
            new_results[key].append(res[key])

    return {key:numpy.array(val) for key, val in new_results.items()}


def _collect(data, t, collectors):
    t1 = time.time()
    sample_dict = dict(time=t)
    for collector in collectors:
        sample = collector(data, t)
        sample_dict.update(sample)
    t_collector = time.time() - t1
    return sample_dict, t_collector


class FixedStepIntegrator:

    def __init__(self, thr, stepper, wiener=None, verbose=True):

        self.thr = thr
        self.verbose = verbose

        self.stepper = stepper.compile(thr)
        if wiener is not None:
            self.noise = True
            self._w = wiener
            self.wiener = wiener.compile(thr)
            self.wiener_double = wiener.double_step().compile(thr)
            self.dW = self.thr.empty_like(wiener.parameter.dW)
            self.dW_state = self.thr.empty_like(wiener.parameter.state)
        else:
            self.noise = False

    def _integrate(self, data, double_step, t_start, dt, steps, samples=None, collectors=None):

        stepper = self.stepper
        noise_dt = dt / 2 if double_step else dt

        if collectors is None:
            collectors = []

        results = []
        step = 0
        sample = 0
        t = t_start

        t_collectors = 0
        t_simulation = time.time()

        if not double_step:
            sample_dict, t_collector = _collect(data, t, collectors)
            results.append(sample_dict)
            t_collectors += t_collector

        if self.noise:
            wiener = self.wiener_double if double_step else self.wiener
            dW_state = self.dW_state
            dW = self.dW
            self.thr.to_device(
                numpy.zeros(wiener.parameter.state.shape, wiener.parameter.state.dtype),
                dest=dW_state)

        if self.verbose:
            title = 'Double step ' if double_step else 'Normal step '
            val_step = steps // 50 if steps >= 50 else 1
            pbar = ProgressBar(
                widgets=[title, Percentage(), Bar(), ETA()], maxval=steps - 1).start()

        for step in _range(steps):

            if self.noise:
                wiener(dW_state, dW, noise_dt)
                stepper(data, data, dW, t, dt)
            else:
                stepper(data, data, t, dt)

            t += dt

            if not double_step:
                if (step + 1) % (steps // samples) == 0:
                    sample_dict, t_collector = _collect(data, t, collectors)
                    results.append(sample_dict)
                    t_collectors += t_collector

            if self.verbose and steps % val_step == 0:
                pbar.update(step)

        if self.verbose:
            pbar.finish()

        t_total = time.time() - t_simulation
        if self.verbose and not double_step:
            print("Collectors time: {f:.1f}%".format(f=t_collectors / t_total * 100))

        if double_step:
            sample_dict, _ = _collect(data, t, collectors)
            return [sample_dict]
        else:
            return results

    def __call__(self, data_dev, t_start, t_end, steps, samples=1, collectors=None):

        if collectors is None:
            collectors = []

        assert steps % samples == 0
        assert steps % 2 == 0
        dt = (t_end - t_start) / steps

        if self.verbose:
            print("Integrating from " + str(t_start) + " to " + str(t_end))

        # double step (to estimate the convergence)
        data_double_dev = self.thr.copy_array(data_dev)
        results_double = self._integrate(
            data_double_dev, True, t_start, dt * 2, steps // 2, collectors=collectors)

        # actual integration
        results = self._integrate(
            data_dev, False, t_start, dt, steps, samples=samples, collectors=collectors)

        # calculate the error (separately for each ensemble)
        # FIXME: should be done on device
        data = data_dev.get()
        data_double = data_double_dev.get()
        data_error = numpy.linalg.norm(data_double - data) / numpy.linalg.norm(data)
        if self.verbose:
            print("Strong error: ", data_error)

        # calculate result errors
        errors = dict(data_strong=data_error)
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

        return _transpose_results(results), errors
