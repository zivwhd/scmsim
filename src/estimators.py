
from dataclasses import dataclass
from datasets import enrich_cause_indexes
from utils import *
from models import *
import pandas as pd
import torch.nn.functional as F


@dataclass
class IPWParams:
    clipping: float = 0.0
    stabilized : float = False
    timed_treatment : bool = True

class MFIPWEstimator:
    def __init__(self, base_name, model, ipw_params):
        self.ipw_params = ipw_params
        self.model = model
        self.base_name = base_name
        
    def estimate(self, uidata, tidx, ridx):
        probs = self.model.probability_matrix()
        watch = uidata.get_watch_matrix()
        timestamp = uidata.get_watch_matrix(timestamps=True).trunc()
        one = torch.ones(1)
        res = {}        
        Wr = watch[:,ridx] * 1.0
        Wt = watch[:,tidx] * 1.0
        Tr = timestamp[:,ridx]
        Tt = timestamp[:,tidx]

        basic_treatment_mask = (Wt > 0.5)
        timed_treatment_mask = ( #(Wt > 0.5) & ((Wr < 0.5) | ((Wr >0.5) & (Tt <= Tr)))
        ( (Wt > 0.5) & (Wr < 0.5) ) | ( (Wt > 0.5) & (Wr > 0.5) & (Tt < Tr)) )
        treatment_prob = probs[:, tidx]

        ipwcfg = []
        for cfg in self.ipw_params:
            cfg_name = (
                "IPW" + 
                (f".clp{cfg.clipping}" if cfg.clipping else "") + 
                (".s" if cfg.stabilized else "") + 
                (".t" if cfg.timed_treatment else ""))
            
            ipwcfg.append((self.base_name + "." + cfg_name, cfg))
            
        for name, cfg in ipwcfg:
            if cfg.timed_treatment:
                treatment_mask = timed_treatment_mask
            else:
                treatment_mask = basic_treatment_mask
            control_mask = ~treatment_mask

            prop1 = torch.maximum(treatment_prob, cfg.clipping * one)
            prop0 = torch.maximum(1-treatment_prob, cfg.clipping * one)

            if cfg.stabilized:
                norm1 = (treatment_mask / prop1).sum(dim=0)
                norm0 = (control_mask / prop0).sum(dim=0)
            else:
                norm1 = Wr.shape[0]
                norm0 = norm1

            Y1 = (treatment_mask * Wr / prop1).sum(dim=0) / norm1
            Y0 = (control_mask * Wr / prop0) .sum(dim=0) / norm0
            ate = (Y1 - Y0).numpy()
            res[name] = ate        
        return pd.DataFrame(res)        

class BasicEstimator:

    def __init__(self):
        pass

    def estimate(self, uidata, tidx, ridx):
        
        watch = uidata.get_watch_matrix()

        one = torch.ones(1)
        res = dict(
            tidx = tidx.numpy(),
            ridx = ridx.numpy()
        )
        
        Wr = watch[:,ridx] * 1.0
        Wt = watch[:,tidx] * 1.0

        ## correlation
        Mt = (Wt*1.0).mean(dim=1, keepdim=True)
        Mr = (Wr*1.0).mean(dim=1, keepdim=True)

        corr = (
            ((Wt - Mt)*(Wr - Mr)).sum(dim=0) / 
            torch.sqrt(((Wt-Mt)**2).sum(dim=0) * ((Wr-Mr)**2).sum(dim=0)))
        
        lift = (Wr * Wt).mean(dim=0) / (Wr.mean(dim=0) * Wt.mean(dim=0))

        treatment_mask = (Wt > 0.5)
        control_mask = ~treatment_mask
        sate = (Wr * treatment_mask).sum(dim=0) / treatment_mask.sum(dim=0) - (Wr * control_mask).sum(dim=0) / control_mask.sum(dim=0)

        return pd.DataFrame(dict(
            CORR = corr.numpy(),
            LIFT = lift.numpy(),
            SATE = sate.numpy()
        ))

class CosineSimilarityEstimator:
    def __init__(self, base_name, model):
        self.model = model
        self.base_name = base_name
        
    @torch.no_grad()
    def estimate(self, uidata, tidx, ridx):
        name = f'{self.base_name}.CosSim'
        return pd.DataFrame({ name : self.cosim(self.model.Q[tidx], self.model.Q[ridx]).numpy() })

    def cosim(self, left, right):
        return (left * right).sum(dim=1) / (left.norm(dim=1) * right.norm(dim=1))