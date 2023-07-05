import torch
import matplotlib.pyplot as plt
from SorscherRNN_cuda import SorscherRNN
from tqdm import tqdm

from dataloader import get_dataloader

if __name__ == "__main__":
    from localconfig import config
    config.read("config")

    model = SorscherRNN() # default parameters are fine
    # move model to GPU if available
    if torch.cuda.is_available():
        model = model.to('cuda')

    dataloader, _ = get_dataloader(config)

    loss_history = [] # we'll use this to plot the loss over time

    num_train_steps = config.training.num_train_steps # number of training steps used by Sorscher et al.
    progress_bar = tqdm(enumerate(dataloader), total=num_train_steps)
    for i, (v, p0, labels, _) in progress_bar:
        loss = model.train_step(v, p0, labels)
        # Update the description and postfix of the progress bar every 100 iterations
        if i % 1 == 0:
            progress_bar.set_description(f"Step {i+1}")
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
            loss_history.append(loss)
        if i % 50 == 0:
            torch.save(model, "model")
        if i > num_train_steps:
            break

    torch.save(model, "model")

    plt.xlabel("Time step")
    plt.ylabel("Loss")
    plt.plot(loss_history)
    plt.show()
