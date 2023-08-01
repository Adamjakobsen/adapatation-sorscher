import torch
import numpy as np
from SorscherRNN_cuda import SorscherRNN
from tqdm import tqdm
import copy

from dataloader import get_dataloader
from logger import Logger
from localconfig import config
import multiprocessing as mp
import sys
import numpy as np
import ratsimulator
from PlaceCells import PlaceCells




config.read("config")
dataloader, dataset = get_dataloader(config)


def grid_search(alpha,beta,gpu_id):
    
    #alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
    #beta = trial.suggest_uniform('beta', 0.0, 1.0)
    
    alpha = alpha
    beta = beta
    weight_decay = config.experiment.weight_decay



    model = SorscherRNN(
        alpha=alpha,
        beta=beta,
        weight_decay=weight_decay,
    )

    device = torch.device(f'cuda:{gpu_id}')
    
    model.to(device)
    #sys.stdout = open(f"./grid_search/grid_search_output{gpu_id}.txt", "a")
    #print("Model is on device:", device)
    num_train_steps = config.training.num_train_steps

    loss_history = []
    decoding_error = []
    mean_dist_list = []
    with open(f"./grid_search/grid_search_output_{gpu_id}_{alpha}_{beta}.txt", "w") as f:
        sys.stdout = f
        print("Model is on device:", device, file=f)


        for i, (v, p0, labels, positions) in enumerate(dataloader):
            #Training
            model.to(device)
            v, p0, labels = v.to(device), p0.to(device), labels.to(device)
            loss = model.train_step(v, p0, labels)
            loss_history.append(loss)
            if i % 100 == 0:
                print(f"Gpu_id {gpu_id};alpha {alpha};beta {beta} Step {i} loss: {loss}",file=f)
                f.flush()

            #Decoding error
            model.eval()
            #model.to("cpu")
            output= model.forward(v,p0)
            #pos_pred= dataset.place_cells.to_euclid(output)
            #mean_dist = torch.mean(torch.abs(positions[:,1:,:]-pos_pred)).item()
            #mean_dist_list.append(mean_dist)
            
            #or just output/labels?
            error = torch.mean(torch.abs(output[:,-1:,:]-labels)).item()
            
            decoding_error.append(error)



            if i > num_train_steps:
                break
    

    
    #Average loss over the last 100 steps
    #average_loss = sum(loss_history[-100:]) / 100
    

    #Save loss history as npy file
    np.save(f"./grid_search/loss_history_{alpha}_{beta}.npy", loss_history)
    np.save(f"./grid_search/decoding_error_{alpha}_{beta}.npy", decoding_error)
    #np.save(f"./grid_search/mean_dist_{alpha}_{beta}.npy", mean_dist_list)
    torch.cuda.empty_cache()
    sys.stdout = sys.__stdout__

if __name__=="__main__":
    alpha_list=np.array([0.001,0.01,0.1,0.5])
    beta_list=np.array([0.001,0.01,0.1,0.5])
    num_gpus = torch.cuda.device_count()

    # Exclude GPU 0
    available_gpus = list(range(1, num_gpus))

    # Create a grid of all combinations of hyperparameters
    param_grid = [(alpha, beta) for alpha in alpha_list for beta in beta_list]

    # Number of processes per GPU
    num_processes_per_gpu = 3

    # Total number of processes
    num_processes = len(available_gpus) * num_processes_per_gpu

    # Create a list of GPU IDs
    gpu_ids = [available_gpus[i % len(available_gpus)] for i in range(num_processes)]


    # Duplicate the GPU IDs list to match the length of the param_grid
    gpu_ids_full = (gpu_ids * num_processes_per_gpu)[:len(param_grid)]

    # Zip together the parameters and GPU IDs
    all_params = [(alpha, beta, gpu_ids_full[i]) for i, (alpha, beta) in enumerate(param_grid)]

    with mp.Pool(processes=num_processes) as pool:
        pool.starmap(grid_search, all_params)




    

    
