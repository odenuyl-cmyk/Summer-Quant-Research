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


def plot_LOB(tcur):
    plt.bar(bid[tcur,:],bidvol[tcur,:],width=0.005)
    plt.bar(ask[tcur,:],askvol[tcur,:],width=0.005)
    plt.title("LOB at t = " + str(t[tcur]) + " seconds")
    plt.xlabel("Price")
    plt.ylabel("Volume")
    plt.show()

# plot_LOB(24730)
# plot_LOB(24731)

#plot the (best) bid/ask prices and midprice
def plot_Mid_Bid_Ask(tcur, t_diff):
    plt.figure()
    plt.plot(t[tcur:(tcur+t_diff)],midprice[tcur:(tcur+t_diff)], label="Midprice")
    plt.plot(t[tcur:(tcur+t_diff)],bid[tcur:(tcur+t_diff),0], label="Best Bid")
    plt.plot(t[tcur:(tcur+t_diff)],ask[tcur:(tcur+t_diff),0], label="Best Ask")
    plt.title('Midprice, Best Ask, and Best Bid')
    plt.ylabel('Price')
    plt.legend()
    plt.xlabel('Time (seconds from trading start)')
    plt.show()

# plot_Mid_Bid_Ask(24730, 1000)

spread = ask[:,0] - bid[:,0]
def plot_Spread(tcur, t_diff):
    plt.figure()
    plt.plot(t[tcur:(tcur+t_diff)],spread[tcur:(tcur+t_diff)], label="Spread")
    plt.title('Spread')
    plt.ylabel('Price')
    plt.legend()
    plt.xlabel('Time (seconds from trading start)')
    plt.show()

# plotSpread(24730, 1000)

def plot_MO_Time_Dist(amt1, amt2):
    plt.figure()
    plt.hist(nMO[:,0]+nMO[:,1],20,(amt1,amt2))
    plt.title(f"MOs in one second interval ({amt1}-{amt2})")
    plt.show()

# plot_MO_Time_Dist(0.1, 5)