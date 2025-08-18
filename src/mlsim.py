import torch
import logging
import pandas as pd


def build_causal_matrix(df, num_items, factor=0.1):
    mat = torch.zeros((num_items, num_items))

    for _, row in df.iterrows():
        t_idx = int(row['treatment_idx']) - 1
        r_idx = int(row['resp_idx']) - 1
        mat[t_idx, r_idx] = row['causal_effect'] * factor

    return mat

def generate_sim_data(base_probs, cmat, iter=10, intervention={}, device='cpu'):
    base_probs = base_probs.to(device)
    probs = (base_probs + 0.0)
    nlcmat = torch.log(1.0-torch.minimum(torch.maximum(cmat, torch.zeros(1)), torch.ones(1) * 0.99)).to(device)
    
    #watched = torch.zeros(probs.shape, device=device)   

    watched = ( torch.rand(base_probs.shape, device=device) < base_probs) * 1.0
    timestamps = watched * torch.rand(watched.shape, device=device)
    
    for  itemid, value in intervention.items():
        watched[:,itemid-1] = value
    
    added = watched
    #selection_mask = torch.ones(probs.shape[0], device=device)
    for idx in range(iter):        
        causal_prob = 1.0-torch.exp(added @ nlcmat) 
        caused =  (torch.rand(causal_prob.shape, device=device) < causal_prob)
        for  itemid, value in intervention.items():
            caused[:,itemid-1] = False        
        added = (caused & (watched < 0.5)) * 1.0
        timestamps += added * (1 + idx +torch.rand(watched.shape, device=device))
        watched += added
        if idx % 10 == 0:
            logging.info(f"[{idx}] - watched:{watched.sum()/1e6:0.2f}M; added:{added.sum()}")
        if added.sum() == 0:
            logging.info("Done")
            break
    return watched, timestamps


def create_pairs_df(watched, timestamps):
    assert watched.shape == timestamps.shape
    num_users, num_items = watched.shape
    base = torch.zeros((num_users,num_items), dtype=torch.int32)
    uidx = torch.arange(num_users).unsqueeze(1) + base
    iidx = torch.arange(num_items).unsqueeze(0) + base
    mask = (watched.flatten() > 0)
    return pd.DataFrame(dict(
        user_id = uidx.flatten()[mask].numpy() + 1,
        item_id = iidx.flatten()[mask].numpy() + 1,
        watched = watched.flatten()[mask].numpy(),
        timestamp = timestamps.flatten()[mask].numpy()
    ))


def generate_ground_truth_estimate(probs, cmat, causes):
    treatment_list = []
    response_list = []
    ate_list = []
    num_items = cmat.shape[0]

    for cidx, cs in enumerate(causes):
        
        control_data, _ = generate_data(probs, cmat, intervention={cs : 0})
        treatment_data, _ = generate_data(probs, cmat, intervention={cs : 1})
        ate = treatment_data.mean(dim=0) - control_data.mean(dim=0)
        
        max_ate = ate[torch.arange(num_items) != (cs - 1)].max()
        logging.info(f"[{cidx}] evaluated cause: {cs}; max-ate:{max_ate}")
        treatment_list += [cs] * num_items
        response_list += list(range(1, 1 + num_items))
        assert ate.shape[0] == num_items
        ate_list += ate.tolist()

    return pd.DataFrame(dict(
        treatment_idx = treatment_list,
        resp_idx = response_list,
        ate = ate_list
    ))


def create_sim_data_samples(paths, name, model, uidata, causal_df, nsamples=1, rewrite=False):
    #pdf = enrich_cause_indexes(pd.read_csv(paths.get_product_csv('MoviesCausalGPT'))), mlm.info)
    probs = model.probability_matrix()
    
    pdf = causal_df
    pdf = pdf[pdf["causal_effect"] >= 0]
    cmat = build_causal_matrix(pdf, uidata.num_items, factor=0.09)

    for idx in range(nsamples):
        out_path = paths.get_product_csv(f'{name}/samples.{idx}')
        if out_path.exists() and not rewrite:
            logging.info(f'{out_path} already exists; skipping')
            continue

        logging.info(f"generating samples {idx}")        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        watched, timestamps = generate_sim_data(probs, cmat)

        df = create_pairs_df(watched, timestamps)
        df.to_csv(out_path, index=False)
        

def create_ground_truth_samples(paths, name, model, uidata, causal_df, idx):
    pdf = causal_df
    pdf = pdf[pdf["causal_effect"] >= 0]

    selected_causes = list(set(pdf[pdf["causal_effect"] > 0]["treatment_idx"]))
    probs = model.probability_matrix()
    cmat = build_causal_matrix(pdf, uidata.num_items, factor=0.09)    
    gtdf = generate_ground_truth_estimate(probs, cmat, selected_causes)    
    
    out_path = paths.get_product_csv(f'{name}/gt.{idx}')
    gtdf.to_csv(out_path, index=False)
    
    filtered_gtdf = pd.merge(pdf, gtdf, on=["treatment_idx", "resp_idx"], how='inner')
    filtered_out_path = paths.get_product_csv(f'{name}/gt.filtered.{idx}')
    filtered_gtdf.to_csv(out_path, index=False)



