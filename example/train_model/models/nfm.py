import numpy as np
import torch

from layers import UIEmbedding, FeaturesEmbedding, FeaturesLinear
from layers import FactorizationMachine, MultiLayerPerceptron


class NFM(torch.nn.Module):
    def __init__(self, field_dims, ui_field_dims, uf_field_dims, if_field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(embed_dim),
            torch.nn.Dropout(0.2)
        )
        self.u_embedding = UIEmbedding(ui_field_dims[:1], embed_dim)
        self.i_embedding = UIEmbedding(ui_field_dims[1:], embed_dim)
        self.uf_embedding = FeaturesEmbedding(uf_field_dims, embed_dim)
        self.if_embedding = FeaturesEmbedding(if_field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(embed_dim, mlp_dims, 0.2)

    def forward(self, user_item, user_cat, item_cat):

        user = user_item[:, 0].reshape(user_item.shape[0], 1)
        item = user_item[:, 1].reshape(user_item.shape[0], 1)

        all_feature = torch.cat((user, item, user_cat, item_cat), axis=1)
        
        user_emb = self.u_embedding(user)
        item_emb = self.i_embedding(item)
        uf_emb = self.uf_embedding(user_cat)
        if_emb = self.if_embedding(item_cat)

        linear = self.linear(all_feature)

        concat_emb = torch.cat((user_emb, item_emb, uf_emb, if_emb), axis=1)
        
        cross_term = self.fm(concat_emb)
        
        mlp = self.mlp(cross_term)
    
        output = linear + mlp

        return torch.sigmoid(output.squeeze(1))
