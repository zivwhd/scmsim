import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import pandas as pd 
import inspect
import sys, logging


def eval_scm(scm, nsamples, **kwargs):
    values = dict(N = nsamples)
    values.update(kwargs)
    
    for varname, func in scm.items():
        argnames = list(inspect.signature(func).parameters.keys())
        logging.info(f"{varname} : f({argnames})")
        args = { name : values[name] for name in argnames }
        values[varname] =func(**args)
    return values

def reeval_scm_with_do(scm, base, **do):    
    values = {}
    values.update(base)    
    modified = set()
    for varname, func in scm.items():
        if varname in do:
            logging.info(f"DO {varname}")
            func = do[varname]                            

        argnames = list(inspect.signature(func).parameters.keys())                        
        if varname not in do and not modified.intersection(argnames):
            continue
        modified.add(varname)
        args = { name : values[name] for name in argnames }
        logging.info(f"{varname} : f({argnames})")
        values[varname] =func(**args)
    return values


def ground_truth_ate(scm, values, treatment, response):
    logging.info("Calculating ground truth")
    values_0 =  reeval_scm_with_do(scm, values, **{treatment : lambda N: torch.zeros(N)})
    values_1 =  reeval_scm_with_do(scm, values, **{treatment : lambda N: torch.ones(N)})
    return (values_1[response] - values_0[response]).mean()

class CondMeanDiff:

    def __init__(self, treatment, response):
        self.treatment_name = treatment
        self.response_name = response

    def __call__(self, values):
        response = values[self.response_name]
        treatment = values[self.treatment_name] 
        return response[treatment >= 0.5].mean() - response[treatment < 0.5].mean()    


class ATEStratified:
    def __init__(self, treatment, response, strat_func):
        self.treatment_name = treatment
        self.response_name = response
        self.strat_func = strat_func        

    def __call__(self, values):        
        response = values[self.response_name]
        treatment = values[self.treatment_name]

        keys = self.strat_func(values)
        unique_keys =set(keys.tolist())
        ate = 0.0
        total_weight = 0
        for k in unique_keys:        
            is_treatment = (treatment > 0.5)
            treatment_mask = (keys == k) & is_treatment
            control_mask = (keys == k) & (~is_treatment)
            if (treatment_mask.sum() == 0 or control_mask.sum() == 0):
                logging.info(f"skipping group {k}")
                continue
            group_ate = (response[treatment_mask].mean() - response[control_mask].mean())
            group_weight = ((keys == k)*1).sum()
            total_weight += group_weight
            ate = ate + group_ate*group_weight
        ate = ate / total_weight
        return ate


class SplitStratify:
    def __init__(self, varname, splits):
        self.varname = varname
        self.splits = torch.Tensor(splits)

    def __call__(self, values):
        vals = values[self.varname]
        keys = torch.zeros(vals.shape)
        for bar in self.splits:
            keys += (vals < bar )
        return keys

class QuantStratify:
    def __init__(self, varname, ngroups):
        self.varname = varname
        self.ngroups = ngroups

    def __call__(self, values):
        vals = values[self.varname]
        keys = torch.zeros(vals.shape)
        quantiles = torch.quantile(vals, torch.arange(self.ngroups)[1:]/float(self.ngroups))
        for q in quantiles:
            keys += (vals <  q)
        return keys

def enrich_propensity(values, target_name, covariate_names, propensity_name="propensity", model=None):
    # Convert target to numpy
    y = values[target_name].cpu().numpy()
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.flatten()

    # Concatenate covariates (converted to numpy)
    X = torch.stack([values[name] for name in covariate_names]).numpy().transpose()    
    # Create and fit sklearn logistic regression
    if model is None:
        model = LogisticRegression(max_iter=1000)
    logging.info(f"fitting model for propensity: '{propensity_name}'")
    model.fit(X, y)

    # Predict probabilities (for positive class)
    probs = model.predict_proba(X)[:, 1]

    # Convert back to torch tensor
    probs_tensor = torch.from_numpy(probs).float()
    values[propensity_name] = probs_tensor

class ATEPropensity:

    def __init__(self, treatment, response, propensity):
        self.treatment_name = treatment
        self.response_name = response
        self.propensity_name = propensity

    def __call__(self, values):
        response = values[self.response_name]
        treatment = values[self.treatment_name] 
        prop = values[self.propensity_name]
        
        treatment_mask = (treatment >= 0.5)
        
        return (
            (response[treatment_mask]/prop[treatment_mask]).sum() / (1.0/prop[treatment_mask]).sum() -
            (response[~treatment_mask]/(1-prop[~treatment_mask])).sum() / (1.0/(1-prop[~treatment_mask])).sum()
        )

class ATEImpute:

    def __init__(self, treatment, response, conditions, mgen=None):
        self.treatment_name = treatment
        self.response_name = response
        self.condition_names = conditions
        self.mgen = mgen or (lambda: GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3))
        
    def __call__(self, values):
        response = values[self.response_name]
        treatment = values[self.treatment_name] 
        
        y = response.numpy()

        # Concatenate covariates (converted to numpy)
        X = torch.stack([values[name] for name in [self.treatment_name] + self.condition_names]).numpy().transpose()    
        model = self.mgen()
        logging.info(f"fitting model for imputation")
        model.fit(X, y)

        treatment_mask = (treatment > 0.5)
        treatment_counterfactual_covs = X[(treatment_mask.numpy()), :]        
        treatment_counterfactual_covs[:,0] = 0
        
        treatment_counterfactual_resp = torch.from_numpy(model.predict(treatment_counterfactual_covs)).float()        

        control_mask = ~treatment_mask
        control_counterfactual_covs = X[(control_mask.numpy()), :]
        control_counterfactual_covs[:,0] = 1
        control_counterfactual_resp = torch.from_numpy(model.predict(control_counterfactual_covs)).float()        

        ate = (
            (response[treatment_mask] -  treatment_counterfactual_resp).sum() + 
            (control_counterfactual_resp - response[control_mask]).sum()) / response.numel()
        return ate
        

def apply_filter(values, mask):
    return {k : (v[mask] if type(v) == torch.Tensor else v) for k,v in values.items() }
