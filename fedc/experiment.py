
from pipeline import *
import time, os, sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


def experiment(NUM_CLIENTS = 3, NUM_FEATURES = 10, FL_ROUNDS = 300, confounding_level=1, seed=42, out_path="sim"):
    device = torch.device("cuda" if torch.cuda.is_available() 
                      else "mps" if torch.backends.mps.is_available() 
                      else "cpu")

    dgp_params = generate_dgp_parameters(
        num_clients=NUM_CLIENTS, 
        num_features=NUM_FEATURES, 
        confounding_level=confounding_level,
        seed=seed,
    )

    print("Generating Training Data...")
    train_loaders = generate_federated_data(
        params=dgp_params, 
        samples_per_client=5000, 
        seed=44 # Train Seed
    )

    print("Generating Testing Data...")
    test_loaders = generate_federated_data(
        params=dgp_params, 
        samples_per_client=1000, 
        seed=200 # Test Seed (Must be different from Train Seed)
    )    


    models = {}

    print("\n2. Training Personalized Propensity Model...")
    global_propensity = PersonalizedPropensityModel(NUM_FEATURES, NUM_CLIENTS)
    trained_propensity = train_federated_model(
        global_propensity, train_loaders, epochs=FL_ROUNDS, is_propensity=True, device=device
    )
    trained_propensity.eval()
    trained_propensity = trained_propensity.to(device)

    print("\n3. Training Baseline Outcome Model (Standard FedAvg)...")
    baseline_outcome = GlobalOutcomeModel(NUM_FEATURES)
    baseline_outcome = train_federated_model(
        baseline_outcome, train_loaders, epochs=FL_ROUNDS, is_propensity=False, propensity_model_for_iptw=None, device=device, end_lr = 0.005    
    )    
    models["FED-AVG"] = baseline_outcome

    print("\n4. Training FED-IPTW Outcome Model (Weighted FedAvg)...")
    iptw_outcome = GlobalOutcomeModel(NUM_FEATURES)
    iptw_outcome = train_federated_model(
        iptw_outcome, train_loaders, epochs=FL_ROUNDS, is_propensity=False, propensity_model_for_iptw=trained_propensity, device=device,
        end_lr = 0.005    
    )
    models["FED-IPTW"] = iptw_outcome
    

    print("\n5. Training Glob-FED-IPTW Outcome Model (Weighted FedAvg)...")
    ipwg_outcome = GlobalOutcomeModel(NUM_FEATURES)
    ipwg_outcome = train_federated_model(
        ipwg_outcome, train_loaders, epochs=FL_ROUNDS, is_propensity=False, propensity_model_for_iptw=trained_propensity, glob_adjusted=True, device=device,
        end_lr = 0.005    
    )
    models["FED-IPWG"] = ipwg_outcome



    print("\n5. Training Tar-FED-IPTW Outcome Model (Weighted FedAvg)...")
    tar_outcome = TarnetOutcomeModel(NUM_FEATURES)
    tar_outcome = train_federated_model(
        tar_outcome, train_loaders, epochs=FL_ROUNDS, is_propensity=False, device=device, end_lr = 0.005    
    )
    models["FED-TARNET"] = tar_outcome


    print("\n6. Training Tar-FED-IPTW Outcome Model (Weighted FedAvg)...")
    taripw_outcome = TarnetOutcomeModel(NUM_FEATURES)
    taripw_outcome = train_federated_model(
        taripw_outcome, train_loaders, epochs=FL_ROUNDS, is_propensity=False, propensity_model_for_iptw=trained_propensity, device=device,
        end_lr = 0.005    
    )
    models["FED-TARNET-IPTW"] = taripw_outcome

    print("\n7. Training Tar-FED-IPTW Outcome Model (Weighted FedAvg)...")
    taripwg_outcome = TarnetOutcomeModel(NUM_FEATURES)
    taripwg_outcome = train_federated_model(
        taripwg_outcome, train_loaders, epochs=FL_ROUNDS, is_propensity=False, propensity_model_for_iptw=trained_propensity, device=device, 
        glob_adjusted=True,
        end_lr = 0.005    
    )
    models["FED-TARNET-IPWG"] = taripwg_outcome

    results = []
    desc = f"CLIENTS{NUM_CLIENTS}_FEAT{NUM_FEATURES}_ROUNDS{FL_ROUNDS}_CONF{confounding_level}_SEED{seed}"

    tm = int(time.time())
    for model_name, omodel in models.items():
        mean_b, ci_b = run_monte_carlo_evaluation(omodel, dgp_params)
        results.append(dict(desc=desc, name=model_name, pehe=mean_b, pehe05=ci_b[0], pehe95=ci_b[1]))
    
    for x in results:
        print(x)

    path = os.path.join(out_path, f"{desc}.{tm}.csv")
    pd.DataFrame(results).to_csv(path, index=False)
    print(f"saved csv at {path}")
    #NUM_CLIENTS = 3, NUM_FEATURES = 10, FL_ROUNDS = 300, confounding_level=1, seed=42, out_path="sim"



import argparse
def main():
    parser = argparse.ArgumentParser(description="Run the Federated Learning Causal Inference Experiment")
    
    # Define lowercase arguments with their corresponding types and defaults
    parser.add_argument('--num_clients', type=int, default=3, 
                        help='Number of federated clients (default: 3)')
    
    parser.add_argument('--num_features', type=int, default=10, 
                        help='Number of features in the dataset (default: 10)')
    
    parser.add_argument('--fl_rounds', type=int, default=300, 
                        help='Number of federated learning rounds (default: 300)')
    
    # Using float here as confounding levels are often fractional (e.g., 0.5, 1.0)
    parser.add_argument('--confounding_level', type=float, default=1.0, 
                        help='Level of confounding in the data generation (default: 1.0)')
    
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (default: 42)')
    
    parser.add_argument('--out_path', type=str, default='sim', 
                        help='Directory or prefix to save the outputs/CSVs (default: "sim")')
    
    args = parser.parse_args()
    
    # Map the parsed lowercase arguments back to the specific function parameters
    experiment(
        NUM_CLIENTS=args.num_clients,
        NUM_FEATURES=args.num_features,
        FL_ROUNDS=args.fl_rounds,
        confounding_level=args.confounding_level,
        seed=args.seed,
        out_path=args.out_path
    )

if __name__ == "__main__":
    main()