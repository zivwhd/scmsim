from pathlib import Path
from dataclasses import dataclass
from datasets import enrich_cause_indexes
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import *
from models import *
import pandas as pd
import yaml
import glob
from functools import reduce
import torch.nn.functional as F



def create_estimations(cfg, uidata, method_name, group_name, estimator, reset_ids=False, batch_size=2**12):

    group_path = cfg.get_product_csv(group_name)
    pdf = pd.read_csv(group_path)
    if reset_ids:
        pdf = enrich_cause_indexes(pdf, uidata.info)

    tidx = torch.tensor((pdf["treatment_idx"] - 1).to_numpy())
    ridx = torch.tensor((pdf["resp_idx"] - 1).to_numpy())
    
    nsamples = ridx.shape[0]
    dfs = []

    for from_idx in range(0, nsamples, batch_size):
        to_idx = min(from_idx + batch_size, nsamples)
        part_tidx = tidx[from_idx:to_idx]
        part_ridx = ridx[from_idx:to_idx]
        logging.info(f"processing: {from_idx}:{to_idx} / {nsamples}")
        est = estimator.estimate(uidata, part_tidx, part_ridx)        
        est["treatment_idx"] = (part_tidx + 1).numpy()
        est["resp_idx"] = (part_ridx + 1).numpy()
        dfs.append(est)

    est = pd.concat(dfs, axis=0, ignore_index=True)

    out_path = cfg.estimation_path(uidata.name(), group_name.replace("/","."), method_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    est.to_csv(out_path, index=False)
    return est

def load_all_estimations(cfg, data_name, group_name, merge_type='inner'):
    pattern = cfg.estimation_path(data_name, group_name.replace("/","."), '*')
    paths = glob.glob(str(pattern))
    dfs = [pd.read_csv(x) for x in paths]
    merged = reduce(lambda left, right: pd.merge(left, right, on=["treatment_idx", "resp_idx"], how=merge_type), dfs)
    return merged

def get_causal_gpt_scores(cfg, uidata, group_name = 'MoviesCausalGPT'):
    
    group_path = cfg.get_product_csv(group_name)
    pdf = pd.read_csv(group_path)
    pdf = enrich_cause_indexes(pdf, uidata.info)

    est = load_all_estimations(cfg, uidata.name(), group_name)
    est_cols = [x for x in est.columns if x not in ["treatment_idx","resp_idx"]]
    pdf = pd.merge(pdf, est, on=["treatment_idx", "resp_idx"], how='inner')
    pdf

    res = []
    for cname in est_cols:
        
        pos_pdf = pdf[pdf["causal_effect"] > 0]
        zero_pdf = pdf[pdf["causal_effect"] == 0]
        
        corr_pos = pos_pdf[cname].corr(pos_pdf["causal_effect"])
        corr = pdf[cname].corr(pdf["causal_effect"])
        zero_mse = (zero_pdf[cname]**2).mean()
        res.append(dict(
            name=cname, corr=corr, corr_pos=corr_pos, zero_mse=zero_mse))
        
    return pd.DataFrame(res)

def get_sim_scores(cfg, uidata, group_name):
    
    group_path = cfg.get_product_csv(group_name)
    pdf = pd.read_csv(group_path)    

    est = load_all_estimations(cfg, uidata.name(), group_name)
    est_cols = [x for x in est.columns if x not in ["treatment_idx","resp_idx"]]
    pdf = pd.merge(pdf, est, on=["treatment_idx", "resp_idx"], how='inner')
    pdf

    res = []
    
    for cname in est_cols:        
        mse = lambda df: F.mse_loss(torch.tensor(df['ate'].to_numpy()), 
                                    torch.tensor(np.nan_to_num(df[cname].to_numpy(),0))).numpy()
        res.append(dict(
            name = cname, 
            mse = mse(pdf),
            mse_pos = mse(pdf[pdf["ate"] > 0.1]),
            mse_zero = mse(pdf[pdf["ate"].abs() < 0.03])))
        
    return pd.DataFrame(res)


        



        












