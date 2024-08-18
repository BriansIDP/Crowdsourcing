import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_scheduler
from sklearn.metrics import roc_curve, auc
import copy

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
    def __init__(self, seed, input_size, hidden_size, dropout_prob=0.0):
        super(TwoLayerMLP, self).__init__()
        
        assert seed is not None, "Please provide a seed for reproducibility"
        self.seed = seed
        self.dropout_prob = dropout_prob
        
        # Initialize layers with the generator
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        
        # Initialize weights with the generator
        self._initialize_weights()

    def _initialize_weights(self):
        # not doing self.generators because that gives error when using joblib
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # Initialize weights using the generator
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu', generator=generator)
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu', generator=generator)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # not doing self.generators because that gives error when using joblib
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        # Apply deterministic dropout
        if self.training:
            dropout_mask = (torch.rand(x.shape, generator=generator) > self.dropout_prob).float().to(x.device)
            x = x * dropout_mask / (1 - self.dropout_prob)
        x = self.fc2(x)  # Output without activation for BCEWithLogitsLoss
        return x

# Prepare the Training Function
def train_neural_net(neural_net, x_train: torch.Tensor, y_train: torch.Tensor, 
                     x_val: torch.Tensor, y_val: torch.Tensor, 
                     lr: float=0.001, weight_decay: float=1e-5, patience: int=20,
                     epochs: int=1000, testing: bool=False, loss_fn_type: str='bce'):
    if loss_fn_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_fn_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid loss function type")
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
            neural_net.eval()
            test_outputs = neural_net(x_val)
            loss_val = criterion(test_outputs, y_val).item()
            val_losses.append(loss_val)
            if loss_fn_type == 'bce':
                train_acc = ((outputs > 0).float() == y_train).float().mean().item()
                val_acc = ((test_outputs > 0).float() == y_val).float().mean().item()
                train_accs.append(train_acc)
                val_accs.append(val_acc)

        # Early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            if loss_fn_type == 'bce':
                best_acc = val_acc
            best_neural_net_wts = copy.deepcopy(neural_net.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            # print(f'Early stopping at epoch {epoch + 1}')
            neural_net.load_state_dict(best_neural_net_wts)  # Load the best neural_net weights
            break
    
    neural_net.eval() # Set the model to evaluation mode
    epoch_min = np.argmin(val_losses)
    best_val_loss = val_losses[epoch_min]
    assert np.isclose(best_loss, best_val_loss)
    if loss_fn_type == 'bce':
        best_val_acc = val_accs[epoch_min]
        assert np.isclose(best_acc, best_val_acc)
    best_train_loss = train_losses[epoch_min]
    # with torch.no_grad():
    #     probs_val = torch.sigmoid(neural_net(x_val)).flatten()
    # fpr, tpr, _ = roc_curve(y_val.numpy(), probs_val.numpy())
    # roc_auc = auc(fpr, tpr)
    stats_dict = {
        "best_train_loss": best_train_loss,
        "best_val_loss": best_val_loss,
        "epoch_min": epoch_min
    }
    if loss_fn_type == 'bce':
        stats_dict["best_val_acc"] = val_accs[epoch_min]
        stats_dict["best_train_acc"] = train_accs[epoch_min]
    if testing:
        stats_dict["train_losses"] = train_losses
        stats_dict["val_losses"] = val_losses
        if loss_fn_type == 'bce':
            stats_dict["train_accs"] = train_accs
            stats_dict["val_accs"] = val_accs
    return neural_net, stats_dict

# Prepare the Training Function
def train_neural_net_with_loaders(neural_net, train_loader, val_loader,
                     lr: float=0.001, weight_decay: float=1e-5, patience: int=20,
                     epochs: int=10, testing: bool=False, loss_fn_type: str='bce',
                     print_every: int=10, num_warmup_steps: int=0,
                     lr_scheduler_type: str='constant'):
    if loss_fn_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_fn_type == 'mse':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid loss function type")
    optimizer = optim.Adam(neural_net.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = epochs * num_update_steps_per_epoch
    num_warmup_steps = num_warmup_steps * max_train_steps

    lr_scheduler = get_scheduler(
        name=lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Training loop
    best_loss=np.inf
    val_losses = []
    val_accs = []
    optimizer.zero_grad()
    for epoch in range(epochs):
        neural_net.train()
        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            # Forward pass
            outputs = neural_net(x_train)
            loss = criterion(outputs, y_train)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            current_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step()
            optimizer.zero_grad()

            if (batch_idx+1) % print_every == 0:
                print(f"Epoch {epoch} (0-idxed), Batch {batch_idx} (0-idxed), Loss: {loss.item()}, LR: {current_lr:.6f}")

        # Validation
        neural_net.eval()
        with torch.no_grad():
            loss_val = 0
            total_samples = 0
            hits = 0
            for batch_idx, (x_val, y_val) in enumerate(val_loader):
                test_outputs = neural_net(x_val)
                loss_val += criterion(test_outputs, y_val).item()*len(y_val)
                total_samples += len(y_val)
                if loss_fn_type == 'bce':
                    hits += ((test_outputs > 0).float() == y_val).float().sum().item()
            val_acc = hits/total_samples
            loss_val /= total_samples
            val_accs.append(val_acc)
            val_losses.append(loss_val)
            print(f"Epoch {epoch} (0-idxed), Validation Loss: {loss_val}, Validation Accuracy: {val_acc}")

        # Early stopping
        if loss_val < best_loss:
            best_loss = loss_val
            if loss_fn_type == 'bce':
                best_acc = val_acc
            best_neural_net_wts = copy.deepcopy(neural_net.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            # print(f'Early stopping at epoch {epoch + 1}')
            neural_net.load_state_dict(best_neural_net_wts)  # Load the best neural_net weights
            break
    
    neural_net.eval() # Set the model to evaluation mode
    epoch_min = np.argmin(val_losses)
    best_val_loss = val_losses[epoch_min]
    assert np.isclose(best_loss, best_val_loss)
    if loss_fn_type == 'bce':
        best_val_acc = val_accs[epoch_min]
        assert np.isclose(best_acc, best_val_acc)
    # with torch.no_grad():
    #     probs_val = torch.sigmoid(neural_net(x_val)).flatten()
    # fpr, tpr, _ = roc_curve(y_val.numpy(), probs_val.numpy())
    # roc_auc = auc(fpr, tpr)
    stats_dict = {
        "best_val_loss": best_val_loss,
        "epoch_min": epoch_min
    }
    if loss_fn_type == 'bce':
        stats_dict["best_val_acc"] = val_accs[epoch_min]
    if testing:
        stats_dict["val_losses"] = val_losses
        if loss_fn_type == 'bce':
            stats_dict["val_accs"] = val_accs
    return neural_net, stats_dict
