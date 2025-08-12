import torch
from torch import nn
import torch.optim as optim

class MatrixFactorization(nn.Module):
    
    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.n_users    = n_users
        self.n_items    = n_items
        self.n_factors  = n_factors

        # biases + embeddings
        self.mu = torch.nn.Parameter(torch.tensor(0, dtype=torch.float32))  # global bias
        self.b_u = torch.nn.Parameter(torch.zeros(n_users, dtype=torch.float32))
        self.b_i = torch.nn.Parameter(torch.zeros(n_items, dtype=torch.float32))
        self.P   = torch.nn.Parameter(torch.randn(n_users,  n_factors, dtype=torch.float32) * 0.1)
        self.Q   = torch.nn.Parameter(torch.randn(n_items,  n_factors, dtype=torch.float32) * 0.1)

    def forward(self, u_idx, i_idx):
        # returns raw score = logit
        return (
            self.mu
            + self.b_u[u_idx]
            + self.b_i[i_idx]
            + (self.P[u_idx] * self.Q[i_idx]).sum(dim=1)
        )

    @torch.no_grad()
    def probablity_matrix(self):
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
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}; Pos-Loss: {loss_pos/total_pos}; Neg-Loss: {loss_neg/total_neg};"+
              f" Acc: {total_accurate/total_items}; Pos-Acc: {accurate_pos/total_pos}; Neg-Acc: {accurate_neg/total_neg}")
        scheduler.step()
    
    return model