import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from sklearn.metrics import roc_curve, auc
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

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

# # Prepare the Training Function
# def train_neural_net_with_loaders(neural_net, train_loader, val_loader,
#                      lr: float=0.001, weight_decay: float=1e-5, patience: int=20,
#                      epochs: int=10, testing: bool=False, loss_fn_type: str='bce',
#                      print_every: int=10, num_warmup_steps: int=0,
#                      lr_scheduler_type: str='constant'):
#     if loss_fn_type == 'bce':
#         criterion = nn.BCEWithLogitsLoss()
#     elif loss_fn_type == 'mse':
#         criterion = nn.MSELoss()
#     else:
#         raise ValueError("Invalid loss function type")
#     optimizer = optim.Adam(neural_net.parameters(), lr=lr, weight_decay=weight_decay)

#     # Scheduler and math around the number of training steps.
#     num_update_steps_per_epoch = len(train_loader)
#     max_train_steps = epochs * num_update_steps_per_epoch
#     num_warmup_steps = num_warmup_steps * max_train_steps

#     lr_scheduler = get_scheduler(
#         name=lr_scheduler_type,
#         optimizer=optimizer,
#         num_warmup_steps=num_warmup_steps,
#         num_training_steps=max_train_steps,
#     )

#     # Training loop
#     best_loss=np.inf
#     val_losses = []
#     val_accs = []
#     optimizer.zero_grad()
#     for epoch in range(epochs):
#         neural_net.train()
#         for batch_idx, (x_train, y_train) in enumerate(train_loader):
#             # Forward pass
#             outputs = neural_net(x_train)
#             loss = criterion(outputs, y_train)

#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()
#             current_lr = optimizer.param_groups[0]['lr']
#             lr_scheduler.step()
#             optimizer.zero_grad()

#             # if (batch_idx+1) % print_every == 0:
#             #     print(f"Epoch {epoch} (0-idxed), Batch {batch_idx} (0-idxed), Loss: {loss.item()}, LR: {current_lr:.6f}")

#         # Validation
#         neural_net.eval()
#         with torch.no_grad():
#             loss_val = 0
#             total_samples = 0
#             hits = 0
#             for batch_idx, (x_val, y_val) in enumerate(val_loader):
#                 test_outputs = neural_net(x_val)
#                 loss_val += criterion(test_outputs, y_val).item()*len(y_val)
#                 total_samples += len(y_val)
#                 if loss_fn_type == 'bce':
#                     hits += ((test_outputs > 0).float() == y_val).float().sum().item()
#             val_acc = hits/total_samples
#             loss_val /= total_samples
#             val_accs.append(val_acc)
#             val_losses.append(loss_val)
#             # print(f"Epoch {epoch} (0-idxed), Validation Loss: {loss_val}, Validation Accuracy: {val_acc}")

#         # Early stopping
#         if loss_val < best_loss:
#             best_loss = loss_val
#             if loss_fn_type == 'bce':
#                 best_acc = val_acc
#             best_neural_net_wts = copy.deepcopy(neural_net.state_dict())
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1

#         if epochs_no_improve >= patience:
#             # print(f'Early stopping at epoch {epoch + 1}')
#             neural_net.load_state_dict(best_neural_net_wts)  # Load the best neural_net weights
#             break
#     # load the best model
#     neural_net.load_state_dict(best_neural_net_wts)
    
#     neural_net.eval() # Set the model to evaluation mode
#     epoch_min = np.argmin(val_losses)
#     best_val_loss = val_losses[epoch_min]
#     assert np.isclose(best_loss, best_val_loss)
#     if loss_fn_type == 'bce':
#         best_val_acc = val_accs[epoch_min]
#         assert np.isclose(best_acc, best_val_acc)
#     # with torch.no_grad():
#     #     probs_val = torch.sigmoid(neural_net(x_val)).flatten()
#     # fpr, tpr, _ = roc_curve(y_val.numpy(), probs_val.numpy())
#     # roc_auc = auc(fpr, tpr)
#     stats_dict = {
#         "best_val_loss": best_val_loss,
#         "epoch_min": epoch_min
#     }
#     if loss_fn_type == 'bce':
#         stats_dict["best_val_acc"] = val_accs[epoch_min]
#     if testing:
#         stats_dict["val_losses"] = val_losses
#         if loss_fn_type == 'bce':
#             stats_dict["val_accs"] = val_accs
#     return neural_net, stats_dict

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TrainWithLoaders:
    def __init__(self, neural_net, train_loader, val_loader, 
                 lr=0.001, weight_decay=1e-5, patience=20,
                 max_grad_steps=1000, testing=False, loss_fn_type='bce', 
                 lr_scheduler_type='constant', eval_interval=50, 
                 use_joblib_fit=False, use_joblib_multirun=True):
        self.neural_net = neural_net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_grad_steps = max_grad_steps
        self.testing = testing
        self.loss_fn_type = loss_fn_type
        self.lr_scheduler_type = lr_scheduler_type
        self.eval_interval = eval_interval
        self.use_joblib_fit = use_joblib_fit
        self.use_joblib_multirun = use_joblib_multirun
        if use_joblib_fit or use_joblib_multirun:
            # Ensure PyTorch uses only one thread per process
            torch.set_num_threads(1)
    
    def init_opti_scheduler(self):
        if self.loss_fn_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.loss_fn_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError("Invalid loss function type")
        self.optimizer = optim.AdamW(self.neural_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        assert self.lr_scheduler_type in ['constant', 'cosine']
        if self.lr_scheduler_type == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_grad_steps, eta_min=0)
    
    def run(self):
        inf_loader = itertools.cycle(self.train_loader)
        best_loss=np.inf
        val_losses = []
        val_accs = []
        self.optimizer.zero_grad()
        break_flag = False
        epoch_no_improve = 0
        for grad_step in range(self.max_grad_steps):
            x_train, y_train = next(inf_loader)
            self.neural_net.train()
            # Forward pass
            outputs = self.neural_net(x_train)
            loss = self.criterion(outputs, y_train)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.lr_scheduler_type == 'cosine':
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if (grad_step % self.eval_interval == 0) or (grad_step == self.max_grad_steps):
                # Validation
                val_acc, loss_val = self.eval()
                val_accs.append(val_acc)
                val_losses.append(loss_val)
                # print(f"Epoch {epoch} (0-idxed), Validation Loss: {loss_val}, Validation Accuracy: {val_acc}")

                # Early stopping
                out = self.early_stopping(loss_val=loss_val, val_acc=val_acc, 
                                          best_loss=best_loss, 
                                          epoch_no_improve=epoch_no_improve, 
                                          grad_step=grad_step)
                best_loss, best_neural_net_wts, epoch_no_improve, break_flag, best_acc = out
                if break_flag:
                    break
        if not break_flag:
            self.neural_net.load_state_dict(best_neural_net_wts)  # Load the best neural_net weights
        else:
            pass # best weights are already loaded

        self.neural_net.eval() # Set the model to evaluation mode
        grad_step_min_id = np.argmin(val_losses)
        best_val_loss = val_losses[grad_step_min_id]
        assert np.isclose(best_loss, best_val_loss)
        if self.loss_fn_type == 'bce':
            best_val_acc = val_accs[grad_step_min_id]
            assert np.isclose(best_acc, best_val_acc)
        stats_dict = {
            "best_val_loss": best_val_loss,
            "grad_step_min": int(grad_step_min_id*self.eval_interval),
        }
        if self.loss_fn_type == 'bce':
            stats_dict["best_val_acc"] = val_accs[grad_step_min_id]
        if self.testing:
            stats_dict["val_losses"] = val_losses
            if self.loss_fn_type == 'bce':
                stats_dict["val_accs"] = val_accs
        return self.neural_net, stats_dict

    def eval(self):
        self.neural_net.eval()
        with torch.no_grad():
            loss_val = 0
            total_samples = 0
            hits = 0
            for x_val, y_val in self.val_loader:
                test_outputs = self.neural_net(x_val)
                loss_val += self.criterion(test_outputs, y_val).item()*len(y_val)
                total_samples += len(y_val)
                if self.loss_fn_type == 'bce':
                    hits += ((test_outputs > 0).float() == y_val).float().sum().item()
            val_acc = hits/total_samples
            loss_val /= total_samples
            return loss_val, val_acc
    
    def early_stopping(self, loss_val, val_acc, best_loss, 
                       epoch_no_improve, grad_step):
        if loss_val < best_loss:
            best_loss = loss_val
            if self.loss_fn_type == 'bce':
                best_acc = val_acc
            best_neural_net_wts = copy.deepcopy(self.neural_net.state_dict())
            epoch_no_improve = 0
        else:
            epoch_no_improve += 1

        if epoch_no_improve >= self.patience:
            epoch = grad_step // self.eval_interval
            print(f'Early stopping at epoch {epoch + 1}')
            # print(f'Early stopping at gradient step {gradient_steps}')
            self.neural_net.load_state_dict(best_neural_net_wts)  # Load the best neural_net weights
            break_flag = True
        else:
            break_flag = False
        
        return best_loss, best_neural_net_wts, epoch_no_improve, break_flag, best_acc