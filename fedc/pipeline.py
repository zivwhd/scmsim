import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy


import torch
from torch.utils.data import TensorDataset, DataLoader

def generate_dgp_parameters(num_clients=3, num_features=10, seed=42, confounding_level=1.0):
    """
    Generates the static 'rules': universal biology and hospital policies.
    """
    torch.manual_seed(seed)
    
    # Universal biology (True relationships)
    weight_y0 = torch.randn(num_features, 1)
    weight_y1 = weight_y0 + 2.5 # True Average Treatment Effect is roughly +2.5
    
    # Hospital-specific biases (Local Policies)
    hospital_policies = []
    for client_id in range(num_clients):
        # We store each hospital's specific confounding bias vector
        #policy = torch.randn(num_features, 1) + (client_id/num_clients) * pfactor
        policy = weight_y0 + torch.randn(num_features, 1)*0.1
        hospital_policies.append(policy)
        
    return {
        'weight_y0': weight_y0,
        'weight_y1': weight_y1,
        'hospital_policies': hospital_policies,
        'num_clients': num_clients,
        'num_features': num_features,
        'confounding_level' : confounding_level
    }


def generate_federated_data(params, samples_per_client, seed, batch_size=10000):
    """
    Samples patient data based on the fixed DGP parameters.
    """
    # Set the seed so we get distinct but reproducible patients for train/test
    torch.manual_seed(seed) 
        
    client_dataloaders = []
    
    # Unpack the fixed rules
    weight_y0 = params['weight_y0']
    weight_y1 = params['weight_y1']
    hospital_policies = params['hospital_policies']
    num_features = params['num_features']
    confounding_level = params['confounding_level']
    
    for client_id in range(params['num_clients']):
        # 1. Generate patient covariates (X)
        X = torch.randn(samples_per_client, num_features)
        rnd = torch.randn(samples_per_client, 1)

        # 2. Apply the FIXED hospital policy to assign treatment
        policy = hospital_policies[client_id]
        policy_logit = torch.matmul(X, policy)

        policy_logit_final = policy_logit * (confounding_level) + rnd * (1-confounding_level)
        #propensity = torch.sigmoid(torch.matmul(X, policy))*0.6 + 0.2
        propensity = torch.sigmoid(policy_logit_final)*0.8 + 0.1
        T = torch.bernoulli(propensity)

        
        # 3. Calculate outcomes using the FIXED biology rules
        noise = torch.randn(samples_per_client, 1) * 0.1
        #nonlinear_bias = torch.sin(X[:, 0:1] * 2.0) * 2.0        
        nonlinear_bias = 0
        #Xp = ((X > -0.1) + (X > -0.5) + (X > 0) + (X > 0.5) + (X > 1)) * 1.0 -2.0
        Xp=X
        Y0 = torch.matmul(Xp, weight_y0) + nonlinear_bias + noise
        Y1 = torch.matmul(Xp, weight_y1) + nonlinear_bias + noise
        
        Y_factual = T * Y1 + (1 - T) * Y0
        True_ITE = Y1 - Y0
        
        # 4. Package data
        cid_tensor = torch.full((samples_per_client, 1), client_id, dtype=torch.long)
        
        dataset = TensorDataset(X, T, Y_factual, True_ITE, cid_tensor, propensity)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        client_dataloaders.append(dataloader)
        
    return client_dataloaders

# ==========================================
# 2. Model Architectures
# ==========================================
class PersonalizedPropensityModel(nn.Module):
    """Predicts P(T|X, H_c). Uses a shared base and hospital-specific heads."""
    def __init__(self, num_features, num_clients):
        super().__init__()
        self.shared_base = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU()
        )
        # Personalized heads (one for each hospital)
        self.local_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(16, 1), nn.Sigmoid()) for _ in range(num_clients)
        ])
        
    def forward(self, x, client_id):
        base_features = self.shared_base(x)
        # Route the forward pass through the specific hospital's head
        # (Assuming batch belongs to a single client for simplicity)
        cid = client_id[0].item() 
        return self.local_heads[cid](base_features)

class GlobalOutcomeModel(nn.Module):
    """Predicts factual outcome Y based on [X, T]."""
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)

class TarnetOutcomeModel(nn.Module):
    """Predicts factual outcome Y based on [X, T]."""
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features + 1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),            
        )
        self.treatment =  nn.Linear(16, 1)
        self.control = nn.Linear(16, 1)

        
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        rep = self.net(inputs)
        out = self.treatment(rep) * t + self.control(rep)*(1-t)
        return out

# ==========================================
# 3. Modular Federated Training Engine
# ==========================================


def train_federated_model_old(global_model, dataloaders, epochs, 
                          is_propensity=False, propensity_model_for_iptw=None,
                          glob_adjusted=False, device='cpu', start_lr = 0.05, end_lr = 0.005):
    optimizer_lr = 0.01
    
    
    global_model = global_model.to(device)


    ####
    global_E_T = 0.5 
    if propensity_model_for_iptw is not None:
        total_T = sum(batch[1].sum().item() for loader in dataloaders for batch in loader)
        total_samples = sum(batch[1].size(0) for loader in dataloaders for batch in loader)
        global_E_T = total_T / total_samples
    ####

    if glob_adjusted:
        with torch.no_grad():
            all_t = []
            all_ps = []
            for client_id, dataloader in enumerate(dataloaders):
                
                for batch in dataloader:
                    x, t, y_factual, _, cid, _prop = [x.to(device) for x in batch]
                    # 1. Patient-Specific Weight (w_{c,i})
                    mps = propensity_model_for_iptw(x, cid)
                    ps = torch.clamp(mps, 0.01, 0.99)
                    #print(t[0].shape, ps.shape)
                    all_t.append(t.squeeze())
                    all_ps.append(ps.squeeze())
            all_t = torch.concat(all_t)
            all_ps = torch.concat(all_ps)

            treatment_norm = (all_t * (1/all_ps)).mean()
            control_norm = ((1-all_t) * (1/(1-all_ps))).mean()

    loss_track = {}
    for round_num in range(epochs):
        client_weights = []
        
        for client_id, dataloader in enumerate(dataloaders):
            local_model = copy.deepcopy(global_model)
            local_model.train()            

            decay_factor = round_num / max(1, epochs - 1) 
            optimizer_lr = start_lr - (start_lr - end_lr) * decay_factor    

            optimizer = optim.Adam(local_model.parameters(), lr=optimizer_lr)
                        

            for batch in dataloader:                
                x, t, y_factual, _, cid, _prop = [x.to(device) for x in batch]

                optimizer.zero_grad()
                
                if is_propensity:
                    # Learning Propensity (Personalized Local Heads)
                    predictions = local_model(x, cid)
                    loss = nn.BCELoss()(predictions, t.to(device))
                else:
                    # Learning Outcome (Generic Global Model)
                    predictions = local_model(x, t)
                    
                    if propensity_model_for_iptw is not None:
                        with torch.no_grad():
                            # 1. Patient-Specific Weight (w_{c,i})
                            mps = propensity_model_for_iptw(x, cid)
                            #ps = _prop
                            ps = mps
                            #aprop = t * _prop + (1-t)*(1-_prop)
                            #aps =  t * mps + (1-t)*(1-mps)
                            #print(torch.transpose(torch.stack([aprop, aps]),0,1))
                            ps = torch.clamp(ps, 0.01, 0.99)


                            ####  
                            if glob_adjusted:
                                final_weights = ((t / ps) / treatment_norm) + (((1-t) / (1-ps)) / control_norm)
                            else:
                                p_T = t.mean()
                                w_ci = t * (p_T / ps) + (1 - t) * ((1 - p_T) / (1 - ps))
                                
                                # 2. Hospital-Specific Global Decorrelation Weight (w_c)
                                # Get propensity for the 'average' patient at this hospital
                                #x_bar = x.mean(dim=0, keepdim=True)
                                #ps_bar = propensity_model_for_iptw(x_bar, cid).squeeze()
                                #ps_bar = torch.clamp(ps_bar, 0.25, 0.75)
                                #w_c = p_T / ps_bar
                                
                                # Combine: Augment patient weights with global hospital adjustment
                                #final_weights = w_ci #* w_c
                                final_weights = torch.clamp(w_ci, 0.1, 10.0)
                        
                        base_losses = nn.MSELoss(reduction='none')(predictions, y_factual)
                        loss = (base_losses * final_weights).mean()
                    else:
                        loss = nn.MSELoss()(predictions, y_factual)
                
                loss.backward()
                optimizer.step()
                loss_track[client_id] = float(loss.cpu())
            client_weights.append(local_model.state_dict())
        print("### epoch: ", round_num, loss_track)
        # Federated Averaging (FedAvg)
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            if 'local_heads' in key: continue # Keep Propensity heads personalized
            global_dict[key] = sum(cw[key] for cw in client_weights) / len(client_weights)
        global_model.load_state_dict(global_dict)
        
    return global_model



import copy
import torch
import torch.nn as nn
import torch.optim as optim

def train_federated_model(global_model, dataloaders, epochs, is_propensity=False, propensity_model_for_iptw=None,
                          glob_adjusted=False, device='cpu', start_lr = 0.05, end_lr = 0.005):
    optimizer_lr = 0.01
    num_clients = len(dataloaders)
    
    global_model = global_model.to(device)
    # ==========================================
    # 1. PRE-COMPUTATION & INITIALIZATION
    # Calculate aggregation weights exactly once
    # ==========================================
    client_agg_weights = []
    
    if propensity_model_for_iptw is not None and not is_propensity:
        # FED-IPTW: Pre-calculate N_c, E[T], X_bar_c, and w_c
        total_T = 0.0
        total_N = 0
        client_N = []
        client_X_bars = []
        
        # First pass over data to gather static stats
        for dataloader in dataloaders:
            c_T = 0.0
            c_N = 0
            c_X_sum = 0.0
            for batch in dataloader:
                x, t, _, _, _, _prop = [x.to(device) for x in batch]
                c_T += t.sum().item()
                c_N += t.size(0)
                c_X_sum += x.sum(dim=0)
                
            total_T += c_T
            total_N += c_N
            client_N.append(c_N)
            client_X_bars.append(c_X_sum / c_N) # The average patient X_c
            
        global_E_T = total_T / total_N
        
        # Calculate the final aggregation weights (n_c * w_c)
        for cid in range(num_clients):
            x_bar = client_X_bars[cid].unsqueeze(0) # Shape [1, num_features]
            cid_tensor = torch.full((1, 1), cid, dtype=torch.long)
            
            with torch.no_grad():
                e_x_bar = propensity_model_for_iptw(x_bar, cid_tensor).squeeze()
                e_x_bar = torch.clamp(e_x_bar, 0.05, 0.95).item()
                
            w_c = global_E_T / e_x_bar
            client_agg_weights.append(client_N[cid] * w_c)
            
    else:
        # STANDARD FedAvg: Weight purely by sample size (n_c)
        for dataloader in dataloaders:
            c_N = sum(batch[1].size(0) for batch in dataloader)
            client_agg_weights.append(c_N)
            
    # Normalize the aggregation weights so they sum to 1.0
    total_weight = sum(client_agg_weights)
    client_agg_weights = [w / total_weight for w in client_agg_weights]

    # ==========================================
    # 2. FEDERATED TRAINING LOOP
    # ==========================================
    loss_track = {}
    for round_num in range(epochs):
        client_weights = []
        
        for client_id, dataloader in enumerate(dataloaders):
            local_model = copy.deepcopy(global_model)
            local_model.train()
            decay_factor = round_num / max(1, epochs - 1) 
            optimizer_lr = start_lr - (start_lr - end_lr) * decay_factor    
            #optimizer_lr = start_lr * (0.99 ** (decay_factor))
            optimizer = optim.Adam(local_model.parameters(), lr=optimizer_lr)
            
            for batch in dataloader:
                x, t, y_factual, _, cid, _prop = [x.to(device) for x in batch]
                optimizer.zero_grad()
                
                if is_propensity:
                    predictions = local_model(x, cid)
                    loss = nn.BCELoss()(predictions, t)
                else:
                    predictions = local_model(x, t)
                    
                    if propensity_model_for_iptw is not None:
                        # IPTW Outcome Training (Local Decorrelation Only)
                        with torch.no_grad():
                            ps = propensity_model_for_iptw(x, cid)
                            ps = torch.clamp(ps, 0.05, 0.95)
                            p_T = t.mean()
                            w_ci = t * (p_T / ps) + (1 - t) * ((1 - p_T) / (1 - ps))
                            w_ci = torch.clamp(w_ci, 0.1, 10.0)
                        
                        base_losses = nn.MSELoss(reduction='none')(predictions, y_factual)
                        loss = (base_losses * w_ci).mean()
                    else:
                        # Baseline Outcome Training
                        loss = nn.MSELoss()(predictions, y_factual)
                
                loss.backward()
                optimizer.step()
                loss_track[client_id] = float(loss.cpu().detach())
                
            client_weights.append(local_model.state_dict())
            
        # ==========================================
        # 3. SERVER AGGREGATION
        # ==========================================
        print("### epoch: ", round_num, loss_track)
        global_dict = global_model.state_dict()
        for key in global_dict.keys():
            if 'local_heads' in key: 
                continue 
            
            # Fast, static weighted average using our normalized pre-computed weights
            weighted_sum = sum(client_weights[i][key] * client_agg_weights[i] for i in range(num_clients))
            global_dict[key] = weighted_sum
            
        global_model.load_state_dict(global_dict)
        
    return global_model

# ==========================================
# 4. Benchmarking (PEHE Score)
# ==========================================
def calculate_pehe(outcome_model, dataloaders, device='cpu'):
    """Calculates the MSE between the predicted ITE and the True ITE."""
    outcome_model.eval()
    outcome_model = outcome_model.to(device)
    total_squared_error = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for dataloader in dataloaders:
            for batch in dataloader:                
                x, _, _, true_ite, _, _ = [z.to(device) for z in batch]
                
                # Predict Y1 (If Treated)
                t_ones = torch.ones(x.shape[0], 1)
                y1_pred = outcome_model(x, t_ones)
                
                # Predict Y0 (If Control)
                t_zeros = torch.zeros(x.shape[0], 1)
                y0_pred = outcome_model(x, t_zeros)
                
                # Predicted ITE = Y1 - Y0
                pred_ite = y1_pred - y0_pred
                
                total_squared_error += torch.sum((pred_ite - true_ite) ** 2).item()
                total_samples += x.shape[0]
                
    pehe_score = total_squared_error / total_samples
    return pehe_score


import numpy as np
import scipy.stats as stats

def run_monte_carlo_evaluation(outcome_model, dgp_params, num_runs=100):
    pehe_scores = []
    
    print(f"Running {num_runs} evaluations...")
    for i in range(num_runs):
        # Sample a fresh test set for every run
        test_loaders = generate_federated_data(dgp_params, samples_per_client=500, seed=i)
        
        # Calculate PEHE
        score = calculate_pehe(outcome_model, test_loaders)
        pehe_scores.append(score)
        
    # Calculate Mean and 95% Confidence Interval
    mean_pehe = np.mean(pehe_scores)
    sem = stats.sem(pehe_scores) # Standard Error of the Mean
    ci_lower, ci_upper = stats.t.interval(0.95, len(pehe_scores)-1, loc=mean_pehe, scale=sem)
    
    return mean_pehe, (ci_lower, ci_upper)

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

def check_propensity_calibration(propensity_model, test_dataloaders):
    """
    Evaluates how well-calibrated the propensity model is using 
    the Brier Score and a Reliability Diagram.
    """
    propensity_model.eval()
    
    all_true_t = []
    all_pred_p = []
    
    with torch.no_grad():
        for dataloader in test_dataloaders:
            for batch in dataloader:
                # Unpack the batch according to our previous DGP
                x, t, _, _, cid, _ = batch
                
                # Get the predicted probability of treatment (the propensity score)
                pred_p = propensity_model(x, cid).squeeze()
                
                # Store the true treatment flags and the predictions
                all_true_t.extend(t.squeeze().tolist())
                all_pred_p.extend(pred_p.tolist())
                
    all_true_t = np.array(all_true_t)
    all_pred_p = np.array(all_pred_p)
    
    # -----------------------------------------
    # 1. Brier Score Calculation
    # -----------------------------------------
    brier_score = brier_score_loss(all_true_t, all_pred_p)
    print(f"Brier Score (0.0 is perfect, 0.25 is random guessing): {brier_score:.4f}")
    
    # -----------------------------------------
    # 2. Calibration Curve (Reliability Diagram)
    # -----------------------------------------
    # Group the predictions into 10 bins to calculate empirical probabilities
    prob_true, prob_pred = calibration_curve(all_true_t, all_pred_p, n_bins=10)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    
    # Plot the model's actual calibration
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Federated Propensity Model')
    
    # Plot the ideal line (y = x)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    
    plt.title('Propensity Score Calibration (Reliability Diagram)')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Actual Treatments (Positives)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    return brier_score

# ==========================================
# 5. Main Execution Flow
# ==========================================
if __name__ == "__main1__":
    NUM_CLIENTS = 3
    NUM_FEATURES = 10
    FL_ROUNDS = 40
    

    dgp_params = generate_dgp_parameters(
        num_clients=NUM_CLIENTS, 
        num_features=NUM_FEATURES, 
        seed=42
    )

    # 2. Sample 800 patients per hospital for Training
    print("Generating Training Data...")
    train_loaders = generate_federated_data(
        params=dgp_params, 
        samples_per_client=3000, 
        seed=100 # Train Seed
    )

    # 3. Sample 200 DIFFERENT patients per hospital for Testing
    print("Generating Testing Data...")
    test_loaders = generate_federated_data(
        params=dgp_params, 
        samples_per_client=1000, 
        seed=200 # Test Seed (Must be different from Train Seed)
    )    

    #print("1. Generating Federated Synthetic Data...")
    #dataloaders = generate_federated_data(NUM_CLIENTS, samples_per_client=1000, num_features=NUM_FEATURES)
    
    print("\n2. Training Personalized Propensity Model...")
    global_propensity = PersonalizedPropensityModel(NUM_FEATURES, NUM_CLIENTS)
    trained_propensity = train_federated_model(
        global_propensity, train_loaders, epochs=FL_ROUNDS, is_propensity=True
    )
    
    sssssssssss

    print("\n3. Training Baseline Outcome Model (Standard FedAvg)...")
    baseline_outcome = GlobalOutcomeModel(NUM_FEATURES)
    baseline_outcome = train_federated_model(
        baseline_outcome, train_loaders, epochs=FL_ROUNDS, is_propensity=False, propensity_model_for_iptw=None
    )
    
    print("\n4. Training FED-IPTW Outcome Model (Weighted FedAvg)...")
    iptw_outcome = GlobalOutcomeModel(NUM_FEATURES)
    iptw_outcome = train_federated_model(
        iptw_outcome, train_loaders, epochs=FL_ROUNDS, is_propensity=False, propensity_model_for_iptw=trained_propensity
    )
    
    print("\n==========================================")
    print("FINAL BENCHMARK: PEHE SCORE (Lower is Better)")
    print("==========================================")
    pehe_baseline = calculate_pehe(baseline_outcome, test_loaders)
    pehe_iptw = calculate_pehe(iptw_outcome, test_loaders)
    
    print(f"Baseline FedAvg PEHE : {pehe_baseline:.4f}")
    print(f"FED-IPTW PEHE        : {pehe_iptw:.4f}")
    print("==========================================")