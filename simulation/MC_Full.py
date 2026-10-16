import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

path = "data/MSFT_20141121.mat"
data = sio.loadmat(path)

LOB = data['LOB']
t = (np.array(LOB['EventTime'][0][0][:, 0]) - 3600000 * 9.5) * 1e-3
bid = np.array(LOB['BuyPrice'][0][0] * 1e-4)
bidvol = np.array(LOB['BuyVolume'][0][0] * 1.0)
ask = np.array(LOB['SellPrice'][0][0] * 1e-4)
askvol = np.array(LOB['SellVolume'][0][0] * 1.0)

k = 10
T = 499
n_paths = 468

edges = np.linspace(-1.0, 1.0, k + 1)
X = 0.5 * (edges[:-1] + edges[1:])
rho = (bidvol[:, 0] - askvol[:, 0]) / (bidvol[:, 0] + askvol[:, 0])
rho_flat = rho.flatten()
idx_series = np.array([np.argmin(np.abs(X - r)) for r in rho_flat])

def estimate_transition_matrix(idx_series, K):
    P = np.zeros((K, K))
    counts = np.zeros(K, dtype=int)

    for i in range(K):
        mask = (idx_series[:-1] == i)
        counts[i] = mask.sum()
        if counts[i] == 0:
            P[i, i] = 1.0
            continue

        next_states = idx_series[1:][mask]
        state_counts = np.bincount(next_states, minlength=K)
        P[i] = state_counts / state_counts.sum()

    return P, counts

P, empirical_counts = estimate_transition_matrix(idx_series, k)
P /= P.sum(axis=1, keepdims=True)

sim_counts = np.zeros(k, dtype=int)

def simulate(state, steps):

    for _ in range(steps):
        sim_counts[state] += 1
        state = np.random.choice(np.arange(k), p=P[state])

    sim_counts[state] += 1
    return state

state = np.random.randint(k)
for _ in range(n_paths):
    state = simulate(state, T)

print("Data Size:", len(t))
print("Simulation Size:", np.sum(sim_counts))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# empirical histogram
hist_counts, _, bars1 = ax1.hist(idx_series, bins=np.arange(k + 1) - 0.5, rwidth=0.8)

ax1.bar_label(bars1, labels=[f"{int(c)}" for c in hist_counts], padding=2, fontsize=8)

ax1.set_xticks(range(k))
ax1.set_xticklabels([f"{x:.1f}" for x in X])
ax1.set_xlabel("State")
ax1.set_ylabel("Count")
ax1.set_title("Empirical Imbalance Distribution")

# simulated histogram
bars2 = ax2.bar(np.arange(k), sim_counts)

ax2.bar_label(bars2, labels=[str(int(c)) for c in sim_counts], padding=2,fontsize=8)

ax2.set_xticks(range(k))
ax2.set_xticklabels([f"{x:.1f}" for x in X])
ax2.set_xlabel("State")
ax2.set_ylabel("Count")
ax2.set_title("Simulated Distribution (Full Transition Matrix)")

plt.tight_layout()
plt.show()