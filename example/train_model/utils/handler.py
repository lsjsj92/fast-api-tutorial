import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import NCF, NFM


class TorchDatasetHander(Dataset):
    def __init__(self, data, u_i_cols=None, u_cat_cols=None, i_cat_cols=None, output_col=None, non_use_col=None):
        self.n = data.shape[0]
        self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)

        self.u_i_cols = u_i_cols
        self.u_cat_cols = u_cat_cols
        self.i_cat_cols = i_cat_cols
        self.non_use_cols = [
            col for col in data.columns if col not in self.u_i_cols + self.u_cat_cols + self.i_cat_cols + [output_col]
        ]
        
        self.i_cont_cols = [
            col for col in data.columns if col not in self.u_i_cols + self.u_cat_cols + self.i_cat_cols + [output_col] + self.non_use_cols
        ]
        
        self.u_i_X = data[self.u_i_cols].astype(np.int).values
        self.u_cat_X = data[self.u_cat_cols].astype(np.int).values
        self.i_cat_X = data[self.i_cat_cols].astype(np.int).values
        self.i_cont_X = data[self.i_cont_cols].astype(np.float32).values
        
        self.field_dims = np.max(np.concatenate((self.u_i_X, self.u_cat_X, self.i_cat_X), axis=1), axis=0) + 1
        self.ui_field_dims = np.max(self.u_i_X, axis=0) + 1
        self.u_cat_field_dims = np.max(self.u_cat_X, axis=0) + 1
        self.i_cat_field_dims = np.max(self.i_cat_X, axis=0) + 1

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.u_i_X[idx], self.u_cat_X[idx],  self.i_cat_X[idx], self.i_cont_X[idx], self.y[idx]]


class ProjectDatasetHander:

    def get_data(self, data_path):
        dataset = pd.read_csv(data_path)
        torch_dataset = TorchDatasetHander(data=dataset, u_i_cols=self.u_i_cols, u_cat_cols=self.u_cat_cols, 
                                     i_cat_cols=self.i_cat_cols, output_col=self.output_col)
        
        return dataset, torch_dataset
    
    def get_data_loader(self, train_torch_dataset, test_torch_dataset):
        train_dataloader = DataLoader(train_torch_dataset, batch_size=self.batch_size, num_workers=0, shuffle=True)
        test_dataloader = DataLoader(test_torch_dataset, batch_size=self.batch_size, num_workers=0, shuffle=False)
        
        return train_dataloader, test_dataloader


class TrainHandler:
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