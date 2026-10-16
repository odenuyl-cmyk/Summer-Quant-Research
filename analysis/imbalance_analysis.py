import scipy.io as sio  # for loading matlab data
import numpy as np      # for numerical libs
from matplotlib.ticker import FuncFormatter # for custom bar plot labels
import matplotlib.pyplot as plt  # for plotting
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)


### LOAD LOB DATA
path = "data/MSFT_20141103.mat"
data = sio.loadmat(path)
LOB=data['LOB']
t = (np.array((LOB['EventTime'][0][0][:,0]))-3600000*9.5)*1e-3 # time in seconds, measured from NASDAQ opening at 9:30am
bid = np.array(LOB['BuyPrice'][0][0]*1e-4)
bidvol = np.array(LOB['BuyVolume'][0][0]*1.0)
ask = np.array(LOB['SellPrice'][0][0]*1e-4)
askvol = np.array(LOB['SellVolume'][0][0]*1.0)
nMO = np.array(LOB['NumberMO'][0][0]*1.0) # total number of MOs submitted in the given 0.1 second interval
vMO = np.array(LOB['VolumeMO'][0][0]*1.0) # total volume of MOs submitted in the given 0.1 second interval
MO=np.array(LOB['MO'][0][0]*1.0) # information about market orders

dt = t[1]-t[0]
imbalanceValues = ((bidvol[:,0]-askvol[:,0])/(bidvol[:,0] + askvol[:,0]))

k = 10 # number of imbalance states
edges = np.linspace(-1.0, 1.0, k + 1)
stateValues = 0.5 * (edges[:-1] + edges[1:])  # bin centers
imbalanceStates = np.argmin(np.abs(imbalanceValues[:, None] - stateValues[None, :]), axis=1)
roundedImbalance = stateValues[imbalanceStates]

def plot_Raw_Imbalance(tcur=0, t_diff=len(t)):
    plt.hist(imbalanceValues[tcur:tcur + t_diff],50,(-1,1))
    print(np.mean(imbalanceValues))
    plt.title("Imbalance Distribution")
    plt.xlabel("Imbalance Value")
    plt.ylabel("Frequency")
    plt.show()
    plt.hist(imbalanceValues,50,(-1,1))

# plot_Raw_Imbalance()

def plot_Imbalance_Steps():
    imbT0 = np.insert(roundedImbalance, 0, 0)
    imbT1 = np.insert(roundedImbalance, -1, 0)
    imbDiff = np.abs(imbT1 - imbT0)
    plt.hist(imbDiff*(k-1)/2, 20,(0.1,20))
    plt.xlabel("State Difference")
    plt.ylabel("Frequency")
    plt.title("Successive Imbalance State Differences")
    plt.show()

# plot_Imbalance_Steps()

def plot_Next_State_Dist():
    current = roundedImbalance[:-1]
    next_vals = roundedImbalance[1:]
    fig, axes = plt.subplots(2, 5, figsize=(15, 7), sharex=False, sharey=False)
    
    for ax, x_i in zip(axes.ravel(), stateValues):
        next_for_i = next_vals[current == x_i]
        next_for_i = next_for_i[next_for_i != x_i]
        counts = [np.sum(next_for_i == x_j) for x_j in stateValues]
        ax.bar(stateValues, counts, width=(edges[1] - edges[0]) * 0.9)
        ax.set_title(f"$x_t={x_i:.1f}$")
        ax.set_xticks(stateValues)
        ax.tick_params(axis='x', labelrotation=45,labelsize=8)

    fig.supxlabel(r"$x_{t+1}$")
    fig.supylabel("Count")
    fig.suptitle("Distribution of Next Imbalance State Given Current State, Not Including Self-Transitions")
    plt.tight_layout()
    plt.show()

# plot_Next_State_Dist()

def plot_Single_Multi_Step_Transition():
    q1s = []
    q2s = []
    current = roundedImbalance[:-1]
    next_vals = roundedImbalance[1:]

    for x_i in stateValues:
        next_for_i = next_vals[current == x_i]
        curr_idx = np.searchsorted(stateValues, x_i)
        next_idx = np.searchsorted(stateValues, next_for_i)
        stepDiff = next_idx - curr_idx
        q_1 = np.sum(np.abs(stepDiff) == 1) / len(stepDiff)
        q_2 = np.sum(np.abs(stepDiff) >= 2) / len(stepDiff)
        q1s.append(q_1)
        q2s.append(q_2)
        print(f"x = {x_i:.2f}, P(step size = 1) = {q_1}")
        print(f"x = {x_i:.2f}, P(step size >= 2) = {q_2}\n")

    plt.figure()
    plt.plot(stateValues, q1s, marker='o', label='1-step probability')
    plt.plot(stateValues, q2s, marker='o', label='multi-step probability')
    plt.title("Jump Probabilities")
    plt.xlabel("λ_t")
    plt.ylabel("Probability")
    plt.ylim(bottom=0)
    plt.legend()
    plt.show()

# plot_Single_Multi_Step_Transition()

def plot_Expected_Step_Size(abs=True):
    current = roundedImbalance[:-1]
    next_vals = roundedImbalance[1:]
    imbDiff = next_vals - current
    normalizer = (k-1)/2

    exp_imb = []
    abs_exp_imb = []

    for x_i in stateValues:
        diffs = imbDiff[current == x_i]
        diffs = diffs[diffs != 0]
        abs_diffs = np.abs(diffs)
        abs_diffs = abs_diffs[abs_diffs != 0]
        exp_imb.append(np.mean(diffs))
        abs_exp_imb.append(np.mean(abs_diffs))

    exp_imb = np.array(exp_imb)
    abs_exp_imb = np.array(abs_exp_imb)

    exp_imb *= normalizer
    abs_exp_imb *= normalizer

    if (not abs):
        plt.figure()
        plt.plot(stateValues, exp_imb, marker='o')
        plt.axhline(0, color='black', linewidth=1)
        plt.title("E[ΔI | I_t = x & ΔI != 0]")
        plt.xlabel("I_t")
        plt.ylabel("E[ΔI]")
        plt.show()
    else:
        plt.figure()
        plt.plot(stateValues, abs_exp_imb, marker='o')
        plt.title("E[abs(ΔI) | I_t = x & ΔI != 0]")
        plt.xlabel("I_t")
        plt.ylabel("E[abs(ΔI)]")
        plt.ylim(bottom=0)
        plt.show()

# plot_Expected_Step_Size()

def plot_MO_Imbalance_Correlation(h=0.2):
    bMO = []
    sMO = []
    bin_centers = []

    for i in range(10):
        a = -1 + i*h
        mask = (imbalanceValues >= a) & (imbalanceValues < a+h)
        bMO_a = np.sum(nMO[mask, 0])  # total buy MOs
        sMO_a = np.sum(nMO[mask, 1])  # total sell MOs
        bMO.append(bMO_a)
        sMO.append(sMO_a)
        bin_centers.append(a + h/2)

    plt.plot(bin_centers, bMO, marker='o', label='Buy MOs')
    plt.plot(bin_centers, sMO, marker='o', label='Sell MOs')
    plt.xlabel('Imbalance')
    plt.ylabel('Number of MOs')
    plt.legend()
    plt.show()

# plot_MO_Imbalance_Correlation()