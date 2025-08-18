import torch
import pickle, yaml
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
    
def resolve_device(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if type(device) == str:
        return torch.device(device)
    
    return device


def validate_cfg(cfg, path=[]):
    missing = []
    for name, value in cfg.items():
        next_path = path + [name]
        if type(value) == dict:
            missing += validate_cfg(value, next_path)
        elif value is None:
            missing.append(".".join(next_path))
    if not path and missing:
        print("Missing config:")
        for x in missing:
            print(f' - {x}')
    return missing

def read_cfg(path):
    with open(path, "rt") as yfile:
        cfg = yaml.safe_load(yfile)
        validate_cfg(cfg)
        return cfg

