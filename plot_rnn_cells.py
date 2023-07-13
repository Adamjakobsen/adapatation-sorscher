import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import multiimshow
import scipy
from dataloader import get_dataloader

if __name__ == "__main__":
    from localconfig import config
    config.read("config")

    # Model class must be defined somewhere
    model = torch.load("model", map_location=torch.device('cpu'))
    model.eval()

    model.to("cpu")

    dataloader, dataset = get_dataloader(config)

    # number of spatial samples across mini-batch and sequence length
    nsamples = 100000
    recurrent_activities = []
    stacked_positions = []
    for i, (v, p0, _, positions) in enumerate(dataloader):
        recurrent_activities.append(model.g(v, p0).cpu().detach().numpy())
        stacked_positions.append(positions[:,1:].cpu().detach().numpy())
        if i*dataloader.batch_size*dataset.seq_len > nsamples:
            break
    # stack runs in mini-batch dimension
    recurrent_activities = np.concatenate(recurrent_activities, axis=0)
    stacked_positions = np.concatenate(stacked_positions, axis=0)
    print("recurrent_activities =", recurrent_activities.shape)
    print("stacked_positions =", stacked_positions.shape)
    # flatten mini-batch and sequence length dimensions
    recurrent_activities = recurrent_activities.reshape(-1, recurrent_activities.shape[-1])
    stacked_positions = stacked_positions.reshape(-1, stacked_positions.shape[-1])
    print("recurrent_activities =", recurrent_activities.shape)
    print("stacked_positions =", stacked_positions.shape)

    # Now we have positions and the correpsonding recurrent activities for each position
    # We can use this to compute the firing field for some example recurrent cells

    ratemaps = scipy.stats.binned_statistic_2d(*stacked_positions.T, recurrent_activities[:,500:700].T, statistic='mean', bins=50)[0]
    print("ratemaps =", ratemaps.shape)

    multiimshow(ratemaps, figsize=(10,10), normalize=False);
    plt.show()
