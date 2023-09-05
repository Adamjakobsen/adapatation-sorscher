import pickle 
import numpy as np
import matplotlib.pyplot as plt

with open("adapt_a6b4_history", 'rb') as f:
    adapt_a6b4_history = pickle.load(f)

with open("adapt_a5b9_history", 'rb') as f:
    adapt_a5b9_history = pickle.load(f)

with open("adapt_a9b9_history", 'rb') as f:
    adapt_a9b9_history = pickle.load(f)

histories = {
    "Adapt a6b4": adapt_a6b4_history, # 0 3 8
    "Adapt a5b9": adapt_a5b9_history, # 3 6 8
    "Adapt a9b9": adapt_a9b9_history  # 0 3 6
}

indexes = {
    "Adapt a6b4": [0, 3, 8],
    "Adapt a5b9": [3, 6, 8],
    "Adapt a9b9": [0, 3, 6]
}

plt.subplots(3, 3, sharex=True, sharey=False)
for i, key in enumerate(histories.keys()):
    for j, index in enumerate(indexes[key]):
        plt.subplot(3,3,j*3+i+1)
        
        plt.plot(np.array(histories[key]["z"]).T[index], label="z" if i==j==2 else None)
        plt.plot(np.array(histories[key]["v"]).T[index], label="v" if i==j==2 else None)

        if j == 0:
            plt.title(key)

plt.legend()
plt.show()