import torch
import os
import time
import json

class Logger:
    def __init__(self, sub_folder=None):
        path = self.get_path(sub_folder)
        self.path = path
        self.make_experiment_folder(path, sub_folder)

    def save_config(self, config) -> None:
        config.save(self.path + "/config")

    def get_path(self, sub_folder: str = None) -> str:
        path = "./experiments"
        if sub_folder is not None:
            path += "/" + sub_folder

        # Make a unique experiment folder name by the time and date
        name = f"/{time.localtime().tm_mday}-{time.localtime().tm_mon}-"+str(time.localtime().tm_year)[-2:]
        name += f"_{time.localtime().tm_hour}:{time.localtime().tm_min}"

        path += name
        # F.ex: ./experiments/5-7-23_18:44

        return path

    def make_experiment_folder(self, path: str, sub_folder: str = None) -> None:
        if not os.path.isdir("./experiments"):
            os.mkdir("./experiments")
        if sub_folder is not None:
            if not os.path.isdir("./experiments" + "/" + sub_folder):
                os.mkdir("./experiments" + "/" + sub_folder)

        additive = 2
        new_path = path
        while os.path.isdir(new_path):
            new_path = path + "_" + str(additive)
            additive += 1
        
        # We are safe to make the folder
        os.mkdir(new_path)
        print("The folder", new_path, "has been made")
        self.path = new_path

    def save_model(self, model) -> None:
        torch.save(model, self.path + "/model")

    def save_list(self, liste, name) -> None:
        with open(self.path + "/" + name, 'w') as f:
            json.dump(liste, f)