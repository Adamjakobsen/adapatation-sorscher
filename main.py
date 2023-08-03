import torch
import matplotlib.pyplot as plt
from SorscherRNN_cuda import SorscherRNN
from tqdm import tqdm

from dataloader import get_dataloader
from logger import Logger

if __name__ == "__main__":
    from localconfig import config
    config.read("config")

    logger = Logger()
    logger.save_config(config)

    model = SorscherRNN(
        alpha=config.experiment.alpha,
        beta=config.experiment.beta,
        weight_decay=config.experiment.weight_decay,
        energy_reg=config.experiment.energy_reg
    ) 
    # move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # MacOS M1 chips have a specific torch speedup:
        # https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
        model = model.to('mps')

    dataloader, _ = get_dataloader(config)

    loss_history = [] # we'll use this to plot the loss over time

    num_train_steps = config.training.num_train_steps

    print("About to start")
    if config.training.on_cluster:
        progress_bar = enumerate(dataloader)
    else:
        progress_bar = tqdm(enumerate(dataloader), total=num_train_steps)
    
    for i, (v, p0, labels, _) in progress_bar:
        loss = model.train_step(v, p0, labels)
        
        # Plotting and visualization
        if i % config.training.plot_interval == 0:
            if not config.training.on_cluster:
                progress_bar.set_description(f"Step {i+1}")
                progress_bar.set_postfix({"loss": f"{loss:.4f}"})
            else:
                print("Epoch:", i, "Loss:", loss)
            loss_history.append(loss)
        if i % config.training.save_interval == 0:
            logger.save_model(model)

            # Remove this later 
            # The handling of saving loss figure should be moved to logger
            plt.figure()
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.plot(loss_history)
            plt.savefig(logger.path + "/loss_figure" + str(i))
        if i > num_train_steps:
            break

    logger.save_model(model)

    # The handling of saving loss figure should be moved to logger
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.savefig(logger.path + "/loss_figure")
