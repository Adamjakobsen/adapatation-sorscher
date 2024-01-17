import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.dataloader import get_dataloader
from src.logger import Logger
from src.SorscherRNN_cuda import SorscherRNN
from utils import decoding_error


def run(config, sub_folder=None):
    logger = Logger(sub_folder=sub_folder)
    logger.save_config(config)

    model = SorscherRNN(
        alpha=config.experiment.alpha,
        beta=config.experiment.beta,
        weight_decay=config.experiment.weight_decay,
        energy_reg=config.experiment.energy_reg,
        non_negativity=config.experiment.non_negativity,
    )
    # move model to GPU if available
    if torch.cuda.is_available():
        model = model.to("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # MacOS M1 chips have a specific torch speedup:
        # https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
        model = model.to("mps")

    dataloader, dataset = get_dataloader(config)

    loss_history = []  # we'll use this to plot the loss over time
    decoding_error_history = []
    l2_reg_history = []
    l2_reg_energy_history = []

    num_train_steps = config.training.num_train_steps

    print("About to start")
    if config.training.on_cluster:
        progress_bar = enumerate(dataloader)
    else:
        progress_bar = tqdm(enumerate(dataloader), total=num_train_steps)

    for i, (v, p0, labels, positions) in progress_bar:
        loss = model.train_step(v, p0, labels)

        # Plotting and visualization
        if i % config.training.plot_interval == 0:
            if not config.training.on_cluster:
                progress_bar.set_description(f"Step {i+1}")
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})
            else:
                print("Epoch:", i, "Loss:", loss)
            loss_history.append(loss)

            model.eval()
            # Get decodign error
            outputs = model.forward(v, p0, log_softmax=True)
            outputs = outputs.to("cpu")
            dec_err = decoding_error(outputs, positions, dataset)
            decoding_error_history.append(dec_err)

            # Get l2 regs
            l2_reg = model.get_l2_reg().cpu().detach().numpy().tolist()
            l2_reg_history.append(l2_reg)

            l2_reg_energy = model.get_l2_reg_energy().cpu().detach().numpy().tolist()
            l2_reg_energy_history.append(l2_reg_energy)

        if i % config.training.save_interval == 0:
            logger.save_model(model)
        if i > num_train_steps:
            break

    logger.save_model(model)

    logger.save_list(loss_history, "loss_figure.json")
    logger.save_list(decoding_error_history, "decoding_error.json")
    logger.save_list(l2_reg_history, "weight_reg.json")
    logger.save_list(l2_reg_energy_history, "l2_reg_energy.json")

    # The handling of saving loss figure should be moved to logger
    plt.figure()
    plt.xlabel("Training steps")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.xticks(
        np.arange(len(loss_history)),
        np.arange(0, len(loss_history) * config.training.plot_interval, config.training.plot_interval),
    )
    plt.savefig(logger.path + "/loss_figure")

    plt.figure()
    plt.xlabel("Training steps")
    plt.ylabel("Decoding error")
    plt.plot(decoding_error_history)
    plt.xticks(
        np.arange(len(loss_history)),
        np.arange(0, len(loss_history) * config.training.plot_interval, config.training.plot_interval),
    )
    plt.savefig(logger.path + "/decoding_error")

    plt.figure()
    plt.xlabel("Training steps")
    plt.ylabel("Weight squared sum")
    plt.plot(l2_reg_history)
    plt.xticks(
        np.arange(len(loss_history)),
        np.arange(0, len(loss_history) * config.training.plot_interval, config.training.plot_interval),
    )
    plt.savefig(logger.path + "/weight_reg")

    plt.figure()
    plt.xlabel("Training steps")
    plt.ylabel("Energy squared sum")
    plt.plot(l2_reg_energy_history)
    plt.xticks(
        np.arange(len(loss_history)),
        np.arange(0, len(loss_history) * config.training.plot_interval, config.training.plot_interval),
    )
    plt.savefig(logger.path + "/energy_reg")


if __name__ == "__main__":
    from localconfig import config

    config.read("config")

    parser = argparse.ArgumentParser(prog="Main", description="Main runs one run with the current config")
    parser.add_argument("-f", "--folder", default=None)
    args = parser.parse_args()

    run(config, sub_folder=args.folder)
