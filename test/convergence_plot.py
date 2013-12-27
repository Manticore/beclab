import numpy
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('convergence.pickle', 'rb') as f:
    results = pickle.load(f)

labels = ["SS-CD", "RK4IP", "RK46-NL"]
colors = {"SS-CD": 'b', "CD": 'y', "RK4IP": 'r', "RK46-NL": 'g'}

fig = plt.figure()
s = fig.add_subplot(111, xlabel='Steps', ylabel='Convergence')
s.set_xscale('log', basex=10)
s.set_yscale('log', basey=10)
for label in labels:
    step_nums = []
    strong_errors = []
    N_errors = []
    N_diffs = []
    SZ2_errors = []
    for steps in sorted(results[label]):
        result = results[label][steps]
        step_nums.append(steps)
        strong_errors.append(result['strong_error'])
        N_errors.append(result['N_convergence'])
        SZ2_errors.append(result['SZ2_convergence'])
        N_diffs.append(result['N_diff'])
    s.plot(step_nums, strong_errors, label=label + ", strong errors",
        color=colors[label], linestyle='-')
    s.plot(step_nums, N_errors, label=label + ", N errors",
        color=colors[label], linestyle='--')
    s.plot(step_nums, SZ2_errors, label=label + ", Sz^2 errors",
        color=colors[label], linestyle=':')
    #s.plot(step_nums, N_diffs, label=label + ", N diffs",
    #    color=colors[label], linestyle='-.')

step_nums = numpy.array(step_nums)
s.plot(step_nums, 1e6 * (1. / step_nums) ** 2, color='grey', linestyle=':', label='1/dt^2 reference')
s.plot(step_nums, 1e12 * (1. / step_nums) ** 4, color='grey', linestyle='--', label='1/dt^4 reference')

s.legend(fontsize=5, loc='lower left')
fig.savefig('convergence_vals.pdf')


fig = plt.figure()
s = fig.add_subplot(111, xlabel='Time (s)', ylabel='Convergence')
s.set_xscale('log', basex=10)
s.set_yscale('log', basey=10)
for label in labels:
    times = []
    step_nums = []
    strong_errors = []
    N_errors = []
    N_diffs = []
    SZ2_errors = []
    for steps in sorted(results[label]):
        result = results[label][steps]
        times.append(result['time'])
        step_nums.append(steps)
        strong_errors.append(result['strong_error'])
        N_errors.append(result['N_convergence'])
        SZ2_errors.append(result['SZ2_convergence'])
        N_diffs.append(result['N_diff'])

    s.plot(times, strong_errors, label=label + ", strong errors",
        color=colors[label], linestyle='-')
    s.plot(times, N_errors, label=label + ", N errors",
        color=colors[label], linestyle='--')
    s.plot(times, SZ2_errors, label=label + ", Sz^2 errors",
        color=colors[label], linestyle=':')
    #s.plot(step_nums, N_diffs, label=label + ", N diffs",
    #    color=colors[label], linestyle='-.')

s.legend(fontsize=5, loc='lower left')
fig.savefig('convergence_vals_normalized.pdf')


fig = plt.figure()
s = fig.add_subplot(111, xlabel='Steps', ylabel='Time (s)')
s.set_xscale('log', basex=10)
s.set_yscale('log', basey=10)
for label in labels:
    step_nums = []
    times = []
    for steps in sorted(results[label]):
        result = results[label][steps]
        step_nums.append(steps)
        times.append(result['time'])
    s.plot(step_nums, times, label=label)
s.legend(fontsize=5, loc='upper left')
fig.savefig('convergence_times.pdf')
