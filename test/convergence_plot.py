import numpy
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

with open('convergence.pickle', 'rb') as f:
    results = pickle.load(f)

labels = ["SS-CD", "CD", "RK4IP", "RK46-NL"]
colors = {"SS-CD": 'b', "CD": 'r', "RK4IP": 'g', "RK46-NL": 'y'}

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
    s.plot(step_nums, strong_errors, label=plot_label + ", strong errors",
        color=colors[plot_label], linestyle='-')
    s.plot(step_nums, N_errors, label=plot_label + ", N errors",
        color=colors[plot_label], linestyle='--')
    s.plot(step_nums, SZ2_errors, label=plot_label + ", Sz^2 errors",
        color=colors[plot_label], linestyle=':')
    s.plot(step_nums, N_diffs, label=plot_label + ", N diffs",
        color=colors[plot_label], linestyle='-.')

s.legend(fontsize=5, loc='lower left')
fig.savefig('convergence_vals.pdf')


fig = plt.figure()
s = fig.add_subplot(111, xlabel='Steps', ylabel='Time (s)')
s.set_xscale('log', basex=2)
s.set_yscale('log', basey=2)
for label in labels:
    step_nums = []
    times = []
    for steps in sorted(results[label]):
        result = results[label][steps]
        step_nums.append(steps)
        times.append(result['time'])
    s.plot(step_nums, times, label=plot_label)
s.legend(fontsize=5, loc='upper left')
fig.savefig('convergence_times.pdf')
