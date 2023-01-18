import os
import argparse

from tqdm import tqdm
import torch.nn as nn
import torch

from utils import TorchDataset, ProjectDataset
from utils import ProjectConfig, ModelConfig
from models import NCF, NFM


class CTRTrain(ProjectConfig, ProjectDataset, ModelConfig):
    def __init__(self, project_path, model_type):
        ProjectConfig.__init__(self, project_path)
        ModelConfig.__init__(self)
        ProjectDataset.__init__(self)
        self.model_type = model_type
    
    def run(self):
        train_dataset, train_torch_dataset, test_dataset, test_torch_dataset = self.get_train_dataset()
        train_dataloader, test_dataloader = self.get_data_loader(train_torch_dataset, test_torch_dataset)

        model = self.get_model(train_torch_dataset)
        model = self.train_model(model, train_dataloader)
        
        torch.save(model.state_dict(), f"{self.project_path}/models/{self.model_type}.pt")
        torch.save(model, f"{self.project_path}/models/{self.model_type}.pkl")
        print("end")
        
    def get_train_dataset(self):
        train_dataset, train_torch_dataset = self.get_data(self.m1m_train_data)
        test_dataset, test_torch_dataset = self.get_data(self.m1m_test_data)
        
        return train_dataset, train_torch_dataset, test_dataset, test_torch_dataset
    
    def get_model(self, dataset):
        if self.model_type == 'ncf':
            model = NCF(
                            dataset.field_dims, dataset.ui_field_dims, embed_dim=16, mlp_dims=(32, 16), dropout=0.2
                        )
        else:
            model = NFM(
                            dataset.field_dims, dataset.ui_field_dims, dataset.u_cat_field_dims, 
                            dataset.i_cat_field_dims, embed_dim=16, mlp_dims=(16, 24), dropout=0.2
                        )
        
        return model
    
    def train_model(self, model, train_dataloader):
        model.train()
        loss_fn = nn.BCELoss()
        model_optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        for _ in tqdm(range(5), total=len(range(5)), position=0, leave=True):
            for samples in train_dataloader:
                model_optimizer.zero_grad()
                user_item, user_cat, item_cat, _, y = samples
                y_pred = model(user_item, user_cat, item_cat)

                loss = loss_fn(y_pred, y.squeeze(1).to(torch.float))
                loss.backward()
                model_optimizer.step()
        
        return model
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type')
    args = parser.parse_args()
    model_type = args.model_type
    # python train.py --model_type ncf
    path = os.path.abspath(os.getcwd())
    parent_path = "/".join(path.split("/")[:-1])
    train_model = CTRTrain(parent_path, model_type)
    train_model.run()
