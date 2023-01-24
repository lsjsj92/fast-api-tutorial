import os
import argparse

import torch.nn as nn
import torch

from utils import ProjectDatasetHander, TrainHandler
from utils import ProjectConfig, ModelConfig


class CTRTrain(ProjectConfig, ProjectDatasetHander, TrainHandler, ModelConfig):
    def __init__(self, project_path, model_type):
        ProjectConfig.__init__(self, project_path)
        ModelConfig.__init__(self)
        ProjectDatasetHander.__init__(self)
        TrainHandler.__init__(self)
        self.model_type = model_type
    
    def run(self):
        train_dataset, train_torch_dataset, test_dataset, test_torch_dataset = self.get_train_dataset()
        train_dataloader, test_dataloader = self.get_data_loader(train_torch_dataset, test_torch_dataset)

        model = self.get_model(train_torch_dataset)
        model = self.train_model(model, train_dataloader)
        
        torch.save(model.state_dict(), f"{self.project_path}/models/{self.model_type}.pt")
        torch.save(model, f"{self.project_path}/models/{self.model_type}.pkl")
        print("end")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type')
    args = parser.parse_args()
    model_type = args.model_type
    # python train.py --model_type ncf
    path = os.path.abspath(os.getcwd())
    project_path = "/".join(path.split("/")[:-2])
    train_model = CTRTrain(project_path, model_type)
    train_model.run()
