import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
from dataclasses import dataclass, asdict
from typing import NamedTuple, Optional, Dict, Any
import logging, json
from pathlib import Path
import pandas as pd
from utils import *

class MatrixFactorization(nn.Module):
    
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.n_users    = n_users
        self.n_items    = n_items
        self.n_factors  = n_factors

        # biases + embeddings
        self.mu = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))  # global bias
        self.b_u = torch.nn.Parameter(torch.zeros(self.n_users, dtype=torch.float32))
        self.b_i = torch.nn.Parameter(torch.zeros(self.n_items, dtype=torch.float32))
        self.P   = torch.nn.Parameter(torch.randn(self.n_users,  self.n_factors, dtype=torch.float32) * 0.1)
        self.Q   = torch.nn.Parameter(torch.randn(self.n_items,  self.n_factors, dtype=torch.float32) * 0.1)

    def forward(self, u_idx, i_idx):
        # returns raw score = logit
        return (
            self.mu
            + self.b_u[u_idx]
            + self.b_i[i_idx]
            + (self.P[u_idx] * self.Q[i_idx]).sum(dim=1)
        )

    @torch.no_grad()
    def probability_matrix(self):
        return torch.sigmoid(
            (self.P @ self.Q.transpose(0,1)) + self.b_u.unsqueeze(1) + self.b_i.unsqueeze(0) + self.mu)            


def train_model(model, loader, epochs=5, lr=1e-3, device=torch.device('cpu'), gamma=0.95):
    # Dataset and DataLoader
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    bce = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(epochs):
        total_loss = 0.0
        total_items = 0
        loss_pos, loss_neg = 0,0
        total_pos, total_neg = 0,0
        total_accurate = 0
        accurate_pos, accurate_neg = 0, 0

        for user_ids, item_ids, labels in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            labels = labels.to(device)
            total_items += user_ids.shape[0]

            # Forward
            logits = model(user_ids, item_ids)

            with torch.no_grad():
                loss_items = bce(logits.squeeze(), labels)
                loss_pos += (loss_items * (labels > 0.5)).sum()
                loss_neg += (loss_items * (labels < 0.5)).sum()
                total_pos += (labels >0.5).sum()
                total_neg += (labels < 0.5).sum()
                total_accurate += ((logits > 0) == (labels > 0.5)).sum()
                accurate_pos += ((logits > 0) & (labels > 0.5)).sum()
                accurate_neg += ((logits < 0) & (labels < 0.5)).sum()

            loss = criterion(logits.squeeze(), labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * user_ids.size(0)

        avg_loss = total_loss / total_items
        logging.info(
            f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}; Pos-Loss: {loss_pos/total_pos}; Neg-Loss: {loss_neg/total_neg};"+
            f" Acc: {total_accurate/total_items}; Pos-Acc: {accurate_pos/total_pos}; Neg-Acc: {accurate_neg/total_neg}")
        scheduler.step()
    
    return model



@dataclass
class MFTrainParams:
    lr : float = 1e-2 
    #wd : float = 1e-7 ## missing
    #pos_weight : float = 1.0 ## missing
    gamma : float = 0.95
    batch_size : int = 2**12
    n_epochs : int = 5
    shuffle : bool = True


class MatrixFactorizationTrain:

    def __init__(
            self,
            cfg,
            name,
            n_factors = 20,
            train_params=MFTrainParams(),
            version=0):
        
        self.cfg = cfg
        self.name = name        
        self.version = version

        self.n_factors = n_factors
        self.train_params = train_params        

        self.data_name = None
        self.model = None
    
    def fit(self,  uidata, device=None, save=True):

        uidata.verify()
        device = resolve_device(device)
        n_users = uidata.num_users
        n_items = uidata.num_items
        dataset = uidata.get_dataset()
        logging.info(f"fitting model {self.name}; num_items={n_items}; num_users={n_users}; params={self.train_params}; device={device}")
        
        tp = self.train_params
        model = MatrixFactorization(n_users, n_items, self.n_factors)

        self.model = model

        loader = DataLoader(dataset, batch_size=tp.batch_size, shuffle=tp.shuffle)
        model = train_model(model, loader, 
                            epochs=tp.n_epochs, lr=tp.lr,  gamma=tp.gamma, device=device)
        self.model = model
        self.data_name = uidata.name()

        if save:
            self.save()
            

    def save(self):        
        dir_path = self.cfg.model_dir_path(self.data_name, self.name, self.version)
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f'saving model at: {dir_path}')
        metadata = dict(
            n_factors = self.n_factors,
            train_params = self.train_params
        )
        with open(dir_path / 'metadata.json', 'tw') as jsonfile:
            json.dump(metadata, jsonfile, cls=PropJSONEncoder, sort_keys=True, indent=2)

        torch.save(self.model.to(torch.device('cpu')), dir_path / self.cfg.model_filename())
        

def load_model(cfg, data_name, model_name, version=0):
    dir_path = cfg.model_dir_path(data_name, model_name, version=0)
    return torch.load(dir_path / cfg.model_filename())

def model_exists(cfg, data_name, model_name, version=0):
    dir_path = cfg.model_dir_path(data_name, model_name, version=0)
    return (dir_path / cfg.model_filename()).exists()

class IPWEstimator:

    def __init__(self, name, propensity_model, configs):
        self.name = name
        #self.probability_matrix = probability_matrix
        self.propensity_model = propensity_model
        self.configs = configs
    
    def eval_ate(self, loader, cause_id, resp_id):

        ncfg = len(self.configs)
        Y1, Y0, norm1, norm0 = ([0] * ncfg, [0] * ncfg, [0] * ncfg, [0] * ncfg)

        for user_id, treatment, response, treatment_time, response_time in loader:

            treatment_mask = (
                treatment & (~response | (treatment_time < response_time))
            )
            control_mask = ~treatment_mask

            ## PUSH_ASSERT - abstract that away #self.probablity_matrix[user_id-1, cause_id]
            propensity =  self.propensity_model(cause_id, user_id)

            for i, cfg in enumerate(self.configs):

                clipping = torch.ones(1) * cfg.clipping
                prop1 = torch.maximum(propensity, clipping)
                prop0 = torch.maximum(1-propensity, clipping)

                Y1[i] += (response * treatment_mask / prop1).sum()
                Y0[i] += (response * control_mask / prop0).sum()

                if cfg.stabilized:
                    norm1[i] += (1.0*treatment_mask/prop1).sum()
                    norm0[i] += (1.0*control_mask/prop0).sum()
                else:
                    nsamples = user_id.shape[0]
                    norm1[i] += nsamples
                    norm0[i] += nsamples

        values =  [
            ((Y1[i] / norm1[i]) - (Y0[i] / norm0[i])).tolist()
            for i in range(len(self.configs))
        ]
        return values

    def get_names(self):
        names = [f"{self.name}.{self.config_to_name(cfg)}" for cfg in self.configs]
        return names
    
    def config_to_name(self, cfg):        
        return ("IPW" + 
                (f".clp{cfg.clipping}" if cfg.clipping else "") + 
                (".s" if cfg.stabilized else ""))
    


        

class Other:

    def save_models(self, base_model_path):
        Path(base_model_path)
        assert self.model is not None

        torch.save(self.model, Path(base_model_path) / f"model.{self.model_name}.pt")

    def load_models(self, base_model_path):
        path = Path(base_model_path) / f"model.{self.model_name}.pt"
        if path.exists():
            self.model = torch.load(path)
            return True
        return False

    @property
    def model_name(self):
        return f"MF{self.n_factors}.{self.version}.{digest(self.train_params)}"
    @property
    def estimator(self):
        if self._estimator is not None:
            return self._estimator
        
        assert self.model is not None
        probs = self.model.probability_matrix()
        propensity_model = lambda cause_id, user_ids: probs[user_ids-1, cause_id-1]            
        self._estimator = IPWEstimator(self.name, propensity_model, self.ipw_params)
        return self._estimator
    


###############################
# #############################                


