import torch
import pickle
import json, logging
from pathlib import Path
from typing import Any, Dict, NamedTuple
from dataclasses import dataclass, is_dataclass, asdict
import hashlib


class PropJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)

def digest(obj):
     jstr = json.dumps(obj, cls=PropJSONEncoder, sort_keys=True)
     return hashlib.md5(jstr.encode("utf-8")).hexdigest()[0:6]


class Config:
    def __init__(self, results_path, products_path):        
        self.results_path = results_path
        self.products_path = products_path

    def model_dir_path(self, data_name, model_name, version=0):
        return Path(self.results_path) / data_name / 'models' / f"{model_name}.{version}"
    
    def model_filename(self):
        return 'weights.pt'
    
    def estimation_path(self, data_name, group_name, method_name):
        return Path(self.results_path) / data_name / 'estimations' / group_name / method_name

    def get_product_csv(self, product):
        return Path(self.products_path) / f'{product}.csv'
    
    
def resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if type(device) == str:
        return torch.device(device)
    
    return device

