from datasets import CsvLoader, MovieLens100KLoader, MovieLens1MLoader
from models import MatrixFactorizationTrain, MFTrainParams
from pathlib import Path
from copy import deepcopy
import torch

class PathProvider:
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


def get_model_trainer(paths, models_cfg, model_name):
    args = models_cfg[model_name] 
    model_type = args.pop('model_type')
    if model_type == 'MatrixFactorization':
        ctor = MatrixFactorizationTrain
        ctor_args = dict(
            n_factors = args.pop('n_factors'),
            train_params = MFTrainParams(**args.pop('train_params'))
        )
    else:
        raise Exception(f"Unexpected model-type: {model_type}")
    unused_args = set(args.keys())
    assert (not unused_args), "unused model args: {unused_args}"

    trainer = ctor(paths=paths, name=model_name, **args)
    return trainer


def get_uidata_loader(global_cfg, data_name):
    args = deepcopy(global_cfg['datasets'][data_name])
    loader_type = args.pop('loader')
    ctor = globals()[loader_type]
    loader = ctor(**args)
    return loader


def load_model(paths, data_name, model_name, version=0):
    dir_path = paths.model_dir_path(data_name, model_name, version=0)
    return torch.load(dir_path / paths.model_filename())

def model_exists(paths, data_name, model_name, version=0):
    dir_path = paths.model_dir_path(data_name, model_name, version=0)
    return (dir_path / paths.model_filename()).exists()
