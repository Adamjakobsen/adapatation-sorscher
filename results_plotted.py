import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import multiimshow
import scipy
from dataloader import get_dataloader
from utils import decoding_error, load_from_file
import os
import json
from robustness import get_robustness

from IPython import embed

color_blind_friendly = {
    "Blue"   : "#377eb8", 
    "Orange" : "#ff7f00", 
    "Green"  : "#4daf4a",
    "Pink"   : "#f781bf", 
    "Brown"  : "#a65628", 
    "Purple" : "#984ea3",
    "Grey"   : "#999999", 
    "Red"    : "#e41a1c", 
    "Yellow" : "#dede00"
}

colors = {
    "vanilla"    : color_blind_friendly["Blue"],
    "energy"     : color_blind_friendly["Green"],
    "adapt_a6b4" : color_blind_friendly["Purple"],
    "adapt_a5b9" : color_blind_friendly["Orange"],
    "adapt_a9b9" : color_blind_friendly["Red"],
}

def load_json(filename):
    content = None
    with open(filename, "r") as f:
        content = json.load(f)
    return content

def get_all_of(filename, folder, load_method):
    path = "./experiments/"

    all_of = []
    for file in os.listdir(path + folder):
        experiment_dir = path + folder + "/" + file
        if os.path.isdir(experiment_dir):
            content = load_method(experiment_dir + "/" + filename)
            all_of.append(content)

    return all_of

def plot_training_metric_all(key_list, metric_name, metric_name_pretty):
    plt.figure()
    for key in key_list:
        for i, loss_history in enumerate(get_all_of(metric_name + ".json", key, load_json)):
            plt.plot(loss_history, c=colors[key], label=key.capitalize().replace("_", " ") if i == 0 else None)
    plt.ylabel(metric_name_pretty)
    plt.xlabel("Training steps")
    plt.xticks(np.linspace(0,1000,5), np.linspace(0,100000,5).astype(int))
    plt.legend()

def plot_training_metric_std_err(key_list, metric_name, metric_name_pretty):
    data_dict = {}
    for key in key_list:
        data_key = []
        for i, loss_history in enumerate(get_all_of(metric_name + ".json", key, load_json)):
            data_key.append(loss_history)
        
        n = len(data_key)
        data_key = np.array(data_key)
        mean = np.sum(data_key, axis=0) / n
    
        std = np.sqrt(np.sum((data_key - mean)**2, axis=0))
        std_error = std / np.sqrt(n)

        data_dict[key] = [
            mean - std_error, 
            mean, 
            mean + std_error
        ]

    plt.figure()
    ax = plt.gca() 
    for key in key_list:
        ax.fill_between(np.arange(len(data_dict[key][0])), data_dict[key][0], data_dict[key][2], color=colors[key]+"55")
        plt.plot(data_dict[key][1], c=colors[key], label=key.capitalize().replace("_", " "))

    plt.xticks(np.linspace(0,1000,5), np.linspace(0,100000,5).astype(int))
    plt.ylabel(metric_name_pretty)
    plt.xlabel("Training steps")
    plt.legend()

def plot_boxplot(datax, labels, metric_name):
    plt.figure()
    plt.boxplot(datax, labels=[label.capitalize().replace("_", " ") for label in labels])

    for i, xs in enumerate(datax):
        noise = np.random.normal(i+1,0.04,len(xs))
        noise[np.argmax(xs)] = i+1
        noise[np.argmin(xs)] = i+1
        plt.scatter(noise, xs, alpha=0.4, color=colors[labels[i]])
        plt.ylabel(metric_name)

def plot_histogram(metric_data):
    num_keys = len(metric_data.keys())

    plt.subplots(1, num_keys, sharex=False, sharey=True)
    for i, key in enumerate(metric_data.keys()):
        hist, bins = metric_data[key]["histogram"]
        bin_num = len(hist)
        bins = np.linspace(0, bins[-2], bin_num)

        plt.subplot(1, num_keys, i+1)
        if i == 0:
            plt.ylabel("Neurons (%)")
        plt.bar(bins, hist, width=bins[-2] / bin_num)
        plt.xlabel("Activity")
        plt.title(label=key.capitalize().replace("_", " "))

def get_metrics(key_list: list, N_silenced: int = 1, bin_num : int = 30) -> dict:
    from localconfig import config
    config.read("config")
     # Getting the test data
    dataloader, dataset = get_dataloader(config)
    datas = []

    metric_data = {}
    for key in key_list:
        metric_data[key] = {
            "difference": [], 
            "mean": []
        }

    for i, key in enumerate(key_list):
        mean_all_sum = []
        models = get_all_of("model", key, load_from_file)
        for j, model in enumerate(models[:]):
            if j > len(datas)-1:
                datas.append(next(enumerate(dataloader))[1])

            difference, mean, mean_all = get_robustness(model, datas[j], dataset, N_silenced)
            metric_data[key]["difference"].append(difference)
            metric_data[key]["mean"].append(mean)
            mean_all_sum += mean_all.tolist()

        hist, bins = np.histogram(mean_all_sum, bins=bin_num)
        
        metric_data[key]["histogram"] = [(hist * 100) / (4096*len(models)), bins]

    return metric_data

def plot_differences_boxplot(metrics):
    plot_boxplot([metrics[key]["difference"] for key in key_list], key_list, "Times worse than original performance")
    plt.title(str(N_silenced) + " most active neurons killed")

def plot_means(metrics):
    plot_boxplot([metrics[key]["mean"] for key in key_list], key_list, "Mean activity")

def get_weights_for(key_list):
    data = {}
    for i, key in enumerate(key_list):
        data[key] = []
        models = get_all_of("model", key, load_from_file)
        for model in models:
            #W, Wh = model.RNN._flat_weights
            #data[key] += W[np.random.randint(0, 4096, (100))].detach().numpy().ravel().tolist()
            W = model.init_position_encoder.weight.ravel()
            data[key] += W[np.random.randint(0, 4096, (100))].detach().numpy().ravel().tolist()

    return data

def plot_weights(key_list):
    data = get_weights_for(key_list)
    plot_boxplot([data[key] for key in key_list], key_list, "Initial position weight value")

if __name__ == "__main__":
    #plot_training_metric_std_err(colors.keys(), "loss_figure", "Loss")
    key_list = list(colors.keys())
    N_silenced = 1
    metrics = get_metrics(key_list, N_silenced, bin_num = 100)

    #plot_differences_boxplot(metrics)
    plot_histogram(metrics)
    #plot_means(metrics)
    #plot_weights(key_list)
    
    plt.show()