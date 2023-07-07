import torch
import matplotlib.pyplot as plt
from SorscherRNN_cuda import SorscherRNN
from tqdm import tqdm
import multiprocessing
import copy

from dataloader import get_dataloader
from logger import Logger

def evaluate(pair, data) -> float:
    alpha, beta = pair
    from localconfig import config
    config.read("config")

    config.experiment.alpha = alpha
    config.experiment.beta = beta

    logger = Logger()
    logger.save_config(config)

    print("Starting defining alpha", alpha, "and beta", beta)
    model = SorscherRNN(
        alpha=config.experiment.alpha,
        beta=config.experiment.beta,
        weight_decay=config.experiment.weight_decay,
    )
    model = model.to("cpu")
    print("Ending defining alpha", alpha, "and beta", beta)

    loss_history = [] # we'll use this to plot the loss over time

    num_train_steps = config.training.num_train_steps

    print("Starting training alpha", alpha, "and beta", beta)

    for i, (v, p0, labels, _) in enumerate(data):
        loss = model.train_step(v, p0, labels)
        print(alpha)

        if i % config.training.plot_interval == 0:
            loss_history.append(loss)
        if i > num_train_steps:
            break

    print("Ending training alpha", alpha, "and beta", beta)

    # The handling of saving loss figure should be moved to logger
    plt.xlabel("Time step")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.savefig(logger.path + "/loss_figure")

    # You'll need to save all the losses
    return loss_history[-1]

if __name__ == "__main__":
    alpha_beta_pairs = [(0, 0), (1,1), (0, 0), (1,1), (0, 0), (1,1), (0, 0), (1,1), (0, 0), (1,1), (0, 0), (1,1)]

    from localconfig import config
    config.read("config")

    #Set torch seed
    torch.manual_seed(0)
    dataloader, _ = get_dataloader(config)

    data = []
    num_train_steps = config.training.num_train_steps
    progress_bar = enumerate(dataloader)
    for i, d in progress_bar:
        data.append(d)

        if i > num_train_steps:
            break
    
    pool = multiprocessing.Pool(12)
    import time 
    start_time = time.time()
    jobs = [pool.apply_async(evaluate, args=(pair,data)) for pair in alpha_beta_pairs]
    losses = [job.get() for job in jobs]
    end_time = time.time()
    print("It took", end_time - start_time)

    print(losses)
