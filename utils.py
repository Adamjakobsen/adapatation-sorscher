import numpy as np
import matplotlib.pyplot as plt
import torch

def load_from_file(filename):
    # Model class must be defined somewhere in this folder
    # I recommend just copying it here from its experiment folder
    model = torch.load(filename, map_location=torch.device('cpu'))
    model.eval()

    model.to("cpu")

    model.RNN.silenced_neurons = None
    try:
        model.RNN.non_negativity = model.RNN.non_negativity
    except:
        model.RNN.non_negativity = False

    return model

def decoding_error(outputs, positions, dataset):
    output_euclid = dataset.place_cells.to_euclid(outputs[:,:,:], k=3)
    difference = positions[:,1:,:] - output_euclid
    diff_abs = torch.abs(difference)
    mean_dist = torch.mean(diff_abs).item()
    return mean_dist

def multiimshow(zz, figsize=(1,1), normalize=False, add_colorbar=True, **kwargs):
    """
    Parameters:
        zz (np.ndarray): A 3D array of shape (nimages, res, res) representing the images to plot.
        figsize (tuple, optional): The figure size. Defaults to (1,1).
        normalize (bool, optional): If True, normalize the images to a common scale. Defaults to True.
        add_colorbar (bool, optional): If True, add a colorbar for the common scale. Defaults to True.
        **kwargs: Additional keyword arguments to pass to matplotlib.pyplot.imshow.
    """
    # prepare figure
    ncols = int(np.ceil(np.sqrt(zz.shape[0])))
    nrows = int(round(np.sqrt(zz.shape[0])))
    from mpl_toolkits.axes_grid1 import ImageGrid
    fig = plt.figure(figsize=figsize)
    if add_colorbar and normalize:
        grid = ImageGrid(fig, rect=(0,0,1,0.87), nrows_ncols=(nrows, ncols), axes_pad=0.05, cbar_mode='single', cbar_location='right', cbar_pad=0.1, cbar_size='5%')
    else:
        grid = ImageGrid(fig, rect=(0,0,1,0.87), nrows_ncols=(nrows, ncols), axes_pad=0)
    vmin, vmax = (np.nanmin(zz), np.nanmax(zz)) if normalize else (None, None)
    # plot response maps using imshow
    for ax, data in zip(grid, zz):
        im = ax.imshow(data, vmin=vmin, vmax=vmax, **kwargs)
    [ax.axis('off') for ax in grid]
    fig.colorbar(im, cax=grid.cbar_axes[0]) if (normalize and add_colorbar) else None
    return fig, grid.axes_all