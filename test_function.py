import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import multiimshow
import scipy
from dataloader import get_dataloader

if __name__ == "__main__":
    from localconfig import config
    config.read("config")

    # Model class must be defined somewhere in this folder
    # I recommend just copying it here from its experiment folder
    model = torch.load("model", map_location=torch.device('cpu'))
    model.eval()

    model.to("cpu")

    # Getting the test data
    dataloader, dataset = get_dataloader(config)
    velocities, init_pc_positions, labels, positions = dataset[0]

    # Getting the model predictions
    output = model.forward(velocities, init_pc_positions)
    output_euclid = dataset.place_cells.to_euclid(output[0,:,:])

    # Getting the loss 
    mean_dist = np.round(torch.mean(torch.sum(torch.abs(positions[1:] - output_euclid), axis=-1)).item(), 5)

    plt.title("Performance on path integration\nMean distance: " + str(mean_dist))
    # Plotting the euclidean positions
    plt.plot(*positions.T, label="True")
    plt.plot(*output_euclid.T, linestyle="dashed", label="Agent")
    # Marking start and end for agent and True labels
    plt.scatter([positions[-1,0]], [positions[-1,1]], marker="*", c="blue", label="True end")
    plt.scatter([output_euclid[-1,0]], [output_euclid[-1,1]], marker="*", c="orange", label="Agent end")
    plt.scatter([positions[1,0]], [positions[1,1]], marker="o", c="blue", label="True start")
    plt.scatter([output_euclid[0,0]], [output_euclid[0,1]], marker="o", c="orange", label="Agent start")
    
    plt.xticks(np.linspace(0,2.2,10))
    plt.yticks(np.linspace(0,2.2,10))
    plt.legend()
    plt.show()

    