import scipy.io as sio  # for loading matlab data
import numpy as np      # for numerical libs
from matplotlib.ticker import FuncFormatter # for custom bar plot labels
import matplotlib.pyplot as plt  # for plotting
import matplotlib.pylab as pylab
import random
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

### LOAD LOB DATA
path = "data/MSFT_20141121.mat"
data = sio.loadmat(path)
LOB=data['LOB']
t = (np.array((LOB['EventTime'][0][0][:,0]))-3600000*9.5)*1e-3 #time in seconds, measured from NASDAQ opening at 9:30am
bid = np.array(LOB['BuyPrice'][0][0]*1e-4)
bidvol = np.array(LOB['BuyVolume'][0][0]*1.0)
ask = np.array(LOB['SellPrice'][0][0]*1e-4)
askvol = np.array(LOB['SellVolume'][0][0]*1.0)
nMO = np.array(LOB['NumberMO'][0][0]*1.0) #total number of MOs submitted in the given 0.1 second interval
vMO = np.array(LOB['VolumeMO'][0][0]*1.0) #total volume of MOs submitted in the given 0.1 second interval
MO=np.array(LOB['MO'][0][0]*1.0) #information about market orders

dt = t[1]-t[0]
midprice = 0.5*(bid[:,0]+ask[:,0])
spread = ask[:,0] - bid[:,0]

T = 600            # Termination time
k = 10            # Number of imbalance states
s = 0.01          # Bid-Ask spread

#Imbalance Grid from -1.0 to +1.0
edges = np.linspace(-1.0, 1.0, k + 1)
X = 0.5 * (edges[:-1] + edges[1:])

#compute LO imbalance for each time stamp
rho = np.array((bidvol[:,0]-askvol[:,0])/(bidvol[:,0]+askvol[:,0]),ndmin=2).T
rho_flat = rho.flatten()

# Value matrix initialized to 0
v = np.zeros((T + 1, k))
# Policy matrix: 0 = CONTINUE, 1 = STOP
policy = np.zeros((T + 1, k), dtype=int)

def estimate_q(idx_series, K):
    q_down = np.zeros(K)
    q_up = np.zeros(K)
    count = np.zeros(K, dtype=int)

    for i in range(K):
        mask = (idx_series[:-1] == i)
        n = mask.sum()
        count[i] = n
        if n == 0:
            continue
        nxt = idx_series[1:][mask]
        if i > 0:
            q_down[i] = np.mean(nxt < i)
        if i < K:
            q_up[i] = np.mean(nxt > i)
    return q_down, q_up, count

idx_series = np.array([np.argmin(np.abs(X - r)) for r in rho_flat])
q_down, q_up, q_count = estimate_q(idx_series, k)

T = 499
counts = [0] * k
def simulate(state, t):
    counts[state] += 1
    n = random.random()
    if t >= T:
        return state
    if n < q_down[state]:
        return simulate(state - 1, t + 1)
    elif (1 - n) < q_up[state]:
        return simulate(state + 1, t + 1)
    else:
        return simulate(state, t + 1)

state = np.random.randint(0, k)

for i in range(468):
    state = simulate(state, 0)

print("Data Size:", len(t))
print("Simulation Size:", np.sum(counts))
    

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Histogram
n, bins, bars1 = ax1.hist(idx_series, bins=np.arange(k + 1) - 0.5, rwidth=0.8)
ax1.bar_label(bars1, fmt='%d', padding=2, fontsize=8)
ax1.set_xticks(range(k))
ax1.set_xlabel("State")
ax1.set_ylabel("Count")
ax1.set_title("Empirical Imbalance Distribution")

# Bar chart
x = np.arange(len(counts))
bars2 = ax2.bar(x, counts)
ax2.bar_label(bars2, padding=2, fontsize=8)
ax2.set_xlabel("State")
ax2.set_ylabel("Count")
ax2.set_xticks(x)
ax2.set_title("Simulated Distribution (Simple Transition Model)")

plt.tight_layout()
plt.show()