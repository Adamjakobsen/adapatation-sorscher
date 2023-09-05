import matplotlib.pyplot as plt
import numpy as np
from dataloader import get_dataloader
from utils import decoding_error, load_from_file

from IPython import embed

def get_robustness(model, data, dataset, N_silenced):
    velocities, init_pc_positions, _, positions = data

    # Compute mean activity for all neurons
    outputs_rnn = model.g(velocities, init_pc_positions) # Get g activations from model
    mean_activity_rnn = np.mean(outputs_rnn.detach().numpy(), axis=(0,1)) # Mean activity of neurons
    mean_activity_number = np.mean(mean_activity_rnn) # Mean activity total, to return

    # Collect the performance before silencing
    outputs_model = model.forward(velocities, init_pc_positions, log_softmax=True)

    # Silence the neurons
    # Find the N neurons with largest activity
    indexes = np.argpartition(mean_activity_rnn, -N_silenced)[-N_silenced:]
    model.RNN.silenced_neurons = indexes

    # Collecting the performance after silencing 
    outputs_model_silenced = model.forward(velocities, init_pc_positions, log_softmax=True)

    # Get the differebce in performance between silenced and non-silenced
    err_normal = decoding_error(outputs_model, positions, dataset)
    err_silenced = decoding_error(outputs_model_silenced, positions, dataset)
    difference = err_silenced / err_normal

    return difference, mean_activity_number, mean_activity_rnn

def test_robustness():
    from localconfig import config
    config.read("config")

    model_vanilla = load_from_file("model_old_weight")
    model_weight = load_from_file("model_reg_z")
    model_adapt = load_from_file("experiments/adapta9b9/13-8-23_4:9/model")

    # Getting the test data
    dataloader, dataset = get_dataloader(config)
    velocities, init_pc_positions, labels, positions = next(enumerate(dataloader))[1]

    N_silenced = 1
    difference_vanilla, mean_vanilla, mean_rnn_vanilla = get_robustness(model_vanilla, (velocities, init_pc_positions, positions), dataset, N_silenced)
    difference_weight, mean_weight, mean_rnn_weight = get_robustness(model_weight, (velocities, init_pc_positions, positions), dataset, N_silenced)
    difference_adapt, mean_adapt, mean_rnn_adapt = get_robustness(model_adapt, (velocities, init_pc_positions, positions), dataset, N_silenced)

    print("Vanilla difference:", difference_vanilla)
    print("Reg Z difference:", difference_weight)
    print("Adapting difference:", difference_adapt)

    print("Mean activity vanilla:", mean_vanilla)
    print("Mean activity weight:", mean_weight)
    print("Mean activity adapt:", mean_adapt)

    plt.figure()
    plt.subplot(1,3,1)
    plt.hist(mean_rnn_vanilla, bins='auto', label="Vanilla")
    plt.legend()
    plt.subplot(1,3,2)
    plt.hist(mean_rnn_weight, bins='auto', label="Weight")
    plt.legend()
    plt.subplot(1,3,3)
    plt.hist(mean_rnn_adapt, bins='auto', label="Adapt")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    test_robustness()
