from tqdm import tqdm
import torch
import pandas as pd
import copy

# Custom Early Stopper with restore best weights
class EarlyStopping:
    def __init__(
        self,
        patience = 5,
        mode="min",
        restore_best_weights=True
    ):
        
        self.patience = patience
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.best_state_dict = None
        self.counter = 0
        self.early_stop = False
        
        if mode not in ["min", "max"]:
            raise ValueError("mode must be one of 'min' or 'max'")
        
    def _is_improvement(self, current_score):
        if self.best_score is None:
            return True
        
        if self.mode == "min":
            return current_score < self.best_score
        else:
            return current_score > self.best_score
    
    def __call__(self, current_score, model):
        if self._is_improvement(current_score):
            self.best_score = current_score
            self.counter = 0
            if self.restore_best_weights:
                self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_state_dict is not None:
                    model.load_state_dict(self.best_state_dict)
        
def train_step(model, 
               dataloader, 
               loss_fn, 
               optimizer, 
               device):
    model.train()
    
    train_loss, train_acc = 0.0, 0.0
    for X, y in tqdm(dataloader, desc="Training batches", leave=False):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_pred = model(X)
        
        # Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        
        # Zero grads and loss backwards, then step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #y_pred_class
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model,
              dataloader,
              loss_fn,
              device):
    model.eval()
    test_loss, test_acc = 0.0, 0.0
   
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Testing batches", leave=False):
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            
            # Calculate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
            
            # y_pred class
            y_pred_class = torch.argmax(y_pred, dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred)
    # Get final metrics
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def main_trainer(model,
                 train_dataloader,
                 test_dataloader,
                 optimizer,
                 loss_fn,
                 total_epochs: int = 5,
                 device='cpu',
                 early_stopper=None):
    history = {
        'train_acc': [],
        'train_loss': [],
        'test_acc': [],
        'test_loss': []
    }
    for epoch in range(total_epochs):
        # Train Step
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )
        
        # Test Step
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device
        )
        history['train_acc'].append(round(train_acc,4))
        history['train_loss'].append(round(train_loss,4))
        history['test_acc'].append(round(test_acc,4))
        history['test_loss'].append(round(test_loss,4))
        
         # Print results
        print(f"Epoch: {epoch+1}")
        print(f"Train_loss: {train_loss: .4f}, Train_acc: {train_acc: .4f} | Test_loss: {test_loss: .4f}, Test_acc: {test_acc: .4f}")
        print('='*80)
        print()
       
        ## Early Stopper
        if early_stopper is not None:
            early_stopper(test_loss, model)
            if early_stopper.early_stop:
                print("Early stopping triggered.")
                break
            
    df = pd.DataFrame(history)
    return df