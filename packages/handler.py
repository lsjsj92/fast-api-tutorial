import torch

class ModelHandler:
    def load_model(self):
        if self.model_type == 'ncf':
            model = torch.load(f'{self.ncf_path}')
        else:
            model = torch.load(f'{self.nfm_path}')
        return model
        

class DataHandler:
    def check_type(self, check_class, data):
        data = check_class(**data)
        
        return data
