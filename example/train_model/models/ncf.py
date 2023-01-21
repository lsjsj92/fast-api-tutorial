import numpy as np
import torch

from layers import UIEmbedding, MultiLayerPerceptron


class NCF(torch.nn.Module):
    def __init__(self, field_dims, ui_field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.embedding_user_mlp = UIEmbedding(ui_field_dims[:1], embed_dim)
        self.embedding_item_mlp = UIEmbedding(ui_field_dims[1:], embed_dim)
        self.embedding_user_mf = UIEmbedding(ui_field_dims[:1], embed_dim)
        self.embedding_item_mf = UIEmbedding(ui_field_dims[1:], embed_dim)
        self.embed_output_dim = len(ui_field_dims) * embed_dim

        self.mlp = MultiLayerPerceptron(embed_dim * 2, mlp_dims, 0.2, output_layer=False)
        self.fc = torch.nn.Linear(mlp_dims[-1] + embed_dim, 1)
        
    def forward(self, user_item, user_cat=None, item_cat=None):

        user = user_item[:, 0].reshape(user_item.shape[0], 1)
        item = user_item[:, 1].reshape(user_item.shape[0], 1)
        
        user_embedding_mlp = self.embedding_user_mlp(user)
        item_embedding_mlp = self.embedding_item_mlp(item)
        user_embedding_mf = self.embedding_user_mf(user).squeeze(1)
        item_embedding_mf = self.embedding_item_mf(item).squeeze(1)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)

        mlp = self.mlp(mlp_vector.view(-1, self.embed_output_dim))

        x = torch.cat([mf_vector, mlp], dim=1)

        output = self.fc(x).squeeze(1)
        
        return torch.sigmoid(output)
