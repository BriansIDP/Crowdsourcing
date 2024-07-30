import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc

def find_kl_gaussians(mu1, var1, mu2, var2):
    mu1 = np.float32(mu1)
    mu2 = np.float32(mu2)
    var1 = np.float32(var1)
    var2 = np.float32(var2)
    return 0.5*(np.log(var2/var1) - 1 + var1/var2 + (mu1-mu2)**2/var2)

def gaussian_log_likelihood(x: np.ndarray, 
                            mean: np.ndarray, 
                            cov: np.ndarray) -> float:
    N, dim = x.shape
    assert mean.shape == (dim,)
    assert cov.shape == (dim, dim)
    
    # Calculate log determinant of the covariance matrix
    sign, logdet = np.linalg.slogdet(cov)
    if sign != 1:
        raise ValueError("Covariance matrix must be positive definite.")
    
    # Calculate the inverse of the covariance matrix
    inv_cov = np.linalg.inv(cov)
    
    # Compute the log likelihood
    diff = x - mean[None, :]
    # log_likelihood = -0.5 * (logdet + diff.T @ inv_cov @ diff + len(x) * np.log(2 * np.pi))
    log_likelihood = -0.5*logdet*np.ones(N) - 0.5*np.sum(diff @ inv_cov * diff, axis=1) \
        - 0.5*dim*np.log(2*np.pi)
    assert log_likelihood.shape == (N,)
    
    return log_likelihood

class TwoLayerMLP(nn.Module):
    def __init__(self, seed, input_size, hidden_size,):
        super(TwoLayerMLP, self).__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        
        # Initialize layers with the generator
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Initialize weights with the generator
        self._initialize_weights()

    def _initialize_weights(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # Initialize weights using the generator
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu', generator=generator)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu', generator=generator)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = self.fc2(x)  # Output without activation for BCEWithLogitsLoss
        return x

# Prepare the Training Function
def train_neural_net(neural_net, x_train: torch.Tensor, y_train: torch.Tensor, 
                     x_val: torch.Tensor, y_val: torch.Tensor, 
                     lr: float=0.001, weight_decay: float=1e-5, patience: int=20,
                     epochs: int=1000, testing: bool=False):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(neural_net.parameters(), lr=lr, weight_decay=weight_decay)

    # Training loop
    best_loss=np.inf
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    for epoch in range(epochs):
        neural_net.train()
        # Forward pass
        outputs = neural_net(x_train)
        loss = criterion(outputs, y_train)
        train_losses.append(loss.item())

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        with torch.no_grad():
            test_outputs = neural_net(x_val)
            loss_val = criterion(test_outputs, y_val)
            val_losses.append(loss_val.item())
            train_accs.append(((outputs > 0).float() == y_train).float().mean().item())
            val_accs.append(((test_outputs > 0).float() == y_val).float().mean().item())

        # Early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            best_neural_net_wts = neural_net.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            # print(f'Early stopping at epoch {epoch + 1}')
            neural_net.load_state_dict(best_neural_net_wts)  # Load the best neural_net weights
            break
    
    epoch_min = np.argmin(val_losses)
    best_val_loss = val_losses[epoch_min]
    assert np.isclose(best_loss, best_val_loss)
    best_train_loss = train_losses[epoch_min]
    # with torch.no_grad():
    #     probs_val = torch.sigmoid(neural_net(x_val)).flatten()
    # fpr, tpr, _ = roc_curve(y_val.numpy(), probs_val.numpy())
    # roc_auc = auc(fpr, tpr)
    stats_dict = {
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        # "roc_auc": roc_auc,
        "best_train_acc": train_accs[epoch_min],
        "best_val_acc": val_accs[epoch_min],
        "epoch_min": epoch_min
    }
    if testing:
        stats_dict["train_losses"] = train_losses
        stats_dict["val_losses"] = val_losses
        stats_dict["train_accs"] = train_accs
        stats_dict["val_accs"] = val_accs
    return neural_net, stats_dict
