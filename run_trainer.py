import argparse
from torch_trainers.torch_custom_models import *
from torch_trainers.torch_data_prepper import generate_dataloaders, augmented_transform
from torch_trainers.training_loop import main_trainer, EarlyStopping
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
import matplotlib.pyplot as plt
import os

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def parser_args():
    parser = argparse.ArgumentParser(description="Training script")
    
    # Paths
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    
    # Train params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e3)
    
    #system
    parser.add_argument("--num_workers", type=int, default=4)
    

    return parser.parse_args()

if __name__ == "__main__":
    args = parser_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training will be run on: {device}")
    model, transforms = make_vgg16(device=device)
    aug_transform = augmented_transform(mean=transforms.mean, std=transforms.std)
    
    train_loader, test_loader = generate_dataloaders(
        train_path=args.train_dir,
        test_path=args.test_dir,
        train_transform=aug_transform,
        test_transform=transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    optimizer = Adam(params=model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    early_stop = EarlyStopping()
    
    df_hist = main_trainer(
        model=model,
        train_dataloader=train_loader,
        test_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        total_epochs=args.epochs,
        device=device,
        early_stopper=early_stop
    )
    
    out_path = Path(args.out_path)
    out_path.mkdir(exist_ok=True, parents=True)
    df_path = out_path / 'history.csv'
    df_hist.to_csv(df_path, index=False)

    epochs = range(1, len(df_hist)+1)
    #--|| Accuracy Plot ||--
    plt.figure()
    plt.plot(epochs, df_hist["train_acc"], label="train_acc")
    plt.plot(epochs, df_hist['test_acc'], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over epochs")
    plt.legend()
    plt.grid(True)
    
    acc_path = out_path / "accuracy.png"
    plt.savefig(acc_path)
    plt.close()
    
    #--|| Loss Plot ||--    
    plt.figure()
    plt.plot(epochs, df_hist["train_loss"], label="train_loss")
    plt.plot(epochs, df_hist['test_loss'], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.legend()
    plt.grid(True)
    
    loss_path = out_path / "loss.png"
    plt.savefig(loss_path)
    plt.close()
    
    