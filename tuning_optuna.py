import torch
import matplotlib.pyplot as plt
from SorscherRNN_cuda import SorscherRNN
from tqdm import tqdm
import multiprocessing
import copy

from dataloader import get_dataloader
from logger import Logger
import optuna
# The objective function that Optuna will optimize
def objective(trial):
    from localconfig import config
    config.read("config")
    alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
    beta = trial.suggest_uniform('beta', 0.0, 1.0)
    print("alpha:", alpha, "beta:", beta)
    weight_decay = config.experiment.weight_decay
    logger = Logger()
    logger.save_config(config)

    model = SorscherRNN(
        alpha=alpha,
        beta=beta,
        weight_decay=weight_decay,
    )

    gpu_id = torch.cuda.current_device()
    device = torch.device(f'cuda:{gpu_id}')
    model.to(device)
    print("Model is on device:", device)

    dataloader, _ = get_dataloader(config)

    num_train_steps = config.training.num_train_steps

    loss_history = []
    for i, (v, p0, labels, _) in enumerate(dataloader):
        v, p0, labels = v.to(device), p0.to(device), labels.to(device)
        loss = model.train_step(v, p0, labels)
        loss_history.append(loss)
        if i > num_train_steps:
            break
    
    #Average loss over the last 100 steps
    average_loss = sum(loss_history[-100:]) / 100
    

    trial.set_user_attr("loss_history", loss_history)

    return average_loss

# Setup Optuna study
study_name = 'SorscherRNN_study'
study_storage = 'sqlite:///tuning_alpha_beta.db'
study = optuna.create_study(study_name=study_name, storage=study_storage, direction='minimize', load_if_exists=True)

# Optimize the objective function
study.optimize(objective, n_trials=100)
