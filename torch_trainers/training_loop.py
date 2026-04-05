from tqdm.auto import tqdm
import torch

def train_step(model, 
               dataloader, 
               loss_fn, 
               optimizer, 
               device):
    model.train()
    
    train_loss, train_acc = 0.0, 0.0
    loop = tqdm(dataloader, desc="Training batches", leave=False)
    for X, y in loop:
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
    loop = tqdm(dataloader, desc="Testing batches", leave=False)
    
    with torch.inference_mode():
        for X, y in loop:
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
                 device='cpu'):
    
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
        
         # Print results
        print(f"Epoch: {epoch+1}")
        print(f"Train_loss: {train_loss: .4f}, Train_acc: {train_acc: .4f} | Test_loss: {test_loss: .4f}, Test_acc: {test_acc: .4f}")
        print('='*80)
        print()