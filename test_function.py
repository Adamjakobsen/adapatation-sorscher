import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch

from src.dataloader import get_dataloader
from utils import decoding_error, load_from_file, multiimshow

if __name__ == "__main__":
    from localconfig import config

    config.read("config")

    model = load_from_file("experiments/adapt_a5b9/12-8-23_14:37/model")

    # Getting the test data
    dataloader, dataset = get_dataloader(config)
    velocities, init_pc_positions, labels, positions = dataset[0]
    W, Wh = model.RNN._flat_weights
    print(torch.mean(torch.abs(W)))

    # Getting the model predictions
    output = model.forward(velocities, init_pc_positions, log_softmax=True)
    output_euclid = dataset.place_cells.to_euclid(output[0, :, :])

    # Getting the loss
    mean_dist = round(decoding_error(output[None], positions[None], dataset), 4)

    plt.figure()
    plt.title("Performance on path integration\nDecoding error: " + str(mean_dist))
    # Plotting the euclidean positions
    plt.plot(*positions.T, label="True")
    plt.plot(*output_euclid.T, linestyle="dashed", label="Agent")
    # Marking start and end for agent and True labels
    plt.scatter([positions[-1, 0]], [positions[-1, 1]], marker="*", c="blue", label="True end")
    plt.scatter([output_euclid[-1, 0]], [output_euclid[-1, 1]], marker="*", c="orange", label="Agent end")
    plt.scatter([positions[1, 0]], [positions[1, 1]], marker="o", c="blue", label="True start")
    plt.scatter([output_euclid[0, 0]], [output_euclid[0, 1]], marker="o", c="orange", label="Agent start")

    plt.xticks(np.linspace(0, 2.2, 10))
    plt.yticks(np.linspace(0, 2.2, 10))
    plt.legend()

    plt.show()
