import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import numpy as np
import re


# Smoothness factor of the graph
smooth_val = 250


log_dnn = open('training_log_DQN.txt','r')

rewards = []
losses = []
wins = []
epsilons = []
x = []

sns.set(style="darkgrid")
plt.xticks(size=12)
plt.yticks(size=12)   
sns.despine(left=True,bottom=True)
plt.xlabel("Number of Batches")

for line in log_dnn:
    splits = re.split("[:\t\n+]",line)
    rewards.append(float(splits[2]))
    losses.append(float(splits[4]))
    wins.append(float(splits[6]))
    epsilons.append(float(splits[8]))
    x.append(int(splits[0]))

reward = np.asarray(rewards)
losses = np.asarray(losses)
wins = np.asarray(wins)
epsilons = np.asarray(epsilons)
x = np.asarray(x)[:-smooth_val]

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



normalize = False  # Normalizes the values to 0-1 range for ease of plotting all in one
if normalize==True:
    rewards = NormalizeData(rewards)
    losses = NormalizeData(losses)
    wins = NormalizeData(wins)
    epsilons = NormalizeData(epsilons)
    
### Comment the required lines in both plot and legend, and set Normalize = False to get true isolated graphs

## __________________________________________________________________
# Plot all (make sure normalize above is set to True) 

plt.title("Plotting using DQN (normalized)")
l1, = plt.plot(x, smooth(reward,smooth_val)[:-smooth_val], antialiased=True,lw=1.75,label="Reward")
l2, = plt.plot(x, smooth(losses,smooth_val)[:-smooth_val], antialiased=True,lw=1.75,label="Loss")
l3, = plt.plot(x, smooth(wins,smooth_val)[:-smooth_val], antialiased=True,lw=1.75,label="Win Rate")
l4, = plt.plot(x, smooth(epsilons,smooth_val)[:-smooth_val], linestyle="dashed",antialiased=True,lw = 1.75,label="Epsilon")

plt.legend(handles=[l1,l2,l3,l4])
plt.show()

## __________________________________________________________________
## Plot % wins per batch vs batches

plt.figure()
plt.ylabel("% Wins per batch")
plt.xlabel("Number of Batches")
plt.title("Win rate curve using DQN")
Wins = plt.plot(x, smooth(wins,smooth_val)[:-smooth_val], antialiased=True,lw=1.75,label="Win Rate")

plt.show()

## __________________________________________________________________
## Plot Rewards vs batches 

plt.figure()
plt.ylabel("Average Rewards")
plt.xlabel("Number of Batches")
plt.title("Average rewards curve using DQN")
Rewards = plt.plot(x, smooth(rewards,smooth_val)[:-smooth_val], antialiased=True,lw=1.75,label="Rewards")

plt.show()

## __________________________________________________________________
## Plot Loss vs batches 

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Number of Batches")
plt.title("Loss curve using DQN")
Loss = plt.plot(x, smooth(losses,smooth_val)[:-smooth_val], antialiased=True,lw=1.75,label="Loss")

plt.show()

## __________________________________________________________________
## Plot Epsilon vs batches 

plt.figure()
plt.xlabel("Number of Batches")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay using DQN")
Ep = plt.plot(x, smooth(epsilons,smooth_val)[:-smooth_val], antialiased=True,lw=1.75,label="Epsilon")

plt.show()
