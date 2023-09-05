import numpy as np
import matplotlib.pyplot as plt

def relu(num):
    return num * (num > 0)

time_steps = 10
non_negativity = True

Wh = 0.001 # Mean value of hidden weights
W = 0.2 # Mean value of input weights

input = np.ones(time_steps)*0.01 # Mean value of input

def get_history(alpha, beta):
    v = 0
    z_prev = 0.0

    hidden = 0.002 # Mean value of hidden 

    z_history = []
    v_history = []

    for i in range(time_steps):
        z = W * input[i]
        z += Wh * hidden
        z -= beta * v

        # Defining v 
        v = v + alpha * (z_prev - v) # Moving away from original equation, in order to have z affected by previous v
        if non_negativity:
            v = relu(v)

        z_history.append(z)
        v_history.append(v)

        # Activation function
        s_z = relu(z)

        hidden = s_z
        z_prev = z

    return z_history, v_history

alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
betas = [0.1,0.2, 0.3, 0.4,0.5,0.6,0.7, 0.8, 0.9]

fig,ax = plt.subplots(len(betas),len(alphas),sharex=True,sharey=True)

for i, beta in enumerate(betas):
    for j, alpha in enumerate(alphas):
        z_history, v_history = get_history(alpha, beta)
        plt.subplot(len(betas),len(alphas),i*len(alphas)+j+1)
        plt.title("⍺: "+str(round(alpha,2)) +" β: " + str(round(beta, 2)))
        plt.plot(z_history, label="Z")
        plt.plot(v_history, label="V")
        if i == len(betas)-1:
            plt.xlabel("Time steps")
        if j == 0:
            plt.ylabel("Activation")
plt.legend(bbox_to_anchor=(1.05, 1.0))
plt.show()

"""saving_beta = []

for i, alpha in enumerate([0.3]):
    for j, beta in enumerate(betas):
        z_history, v_history = get_history(alpha, beta)
        saving_beta.append(
            np.mean(np.abs(z_history))
        )

plt.plot(betas, saving_beta)
plt.xlabel("beta")
plt.ylabel("|z|")
plt.show()"""