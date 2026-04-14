import pandas as pd
import os
from .torch_data_prepper import generate_dataloaders
from .training_loop import main_trainer
from .torch_custom_models import *
from torchvision import models
import torch.nn as nn
from torch.optim import Adam
import torch
from pathlib import Path

def save_all_results(all_history, all_results, save_dir):
    results_df = pd.DataFrame(all_results)
    final_history = pd.concat(all_history, ignore_index=True)
    results_path = f"{save_dir}/all_results.csv"
    results_header = not os.path.exists(results_path)
    results_df.to_csv(results_path, index=False, mode='a', header=results_header)
    hist_path = f"{save_dir}/all_history.csv"
    hist_header = not os.path.exists(hist_path)
    final_history.to_csv(hist_path, index=False, mode='a', header=hist_header)
    

def get_transforms(model_name):
    match model_name:
        case "resnet18":
            return models.ResNet18_Weights.DEFAULT.transforms()
        case "efficientnet_b0":
            return models.EfficientNet_B0_Weights.DEFAULT.transforms()
        case "vgg16":
            return models.VGG16_Weights.DEFAULT.transforms()
        case "mobilenet_v3_small":
            return models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
        case _:
            raise ValueError("Wrong model provided.")

def build_model(model_name, dropout, device):
    if model_name == "resnet18":
        model, transforms = make_resnet18()
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 5)
        )

    elif model_name == "efficientnet_b0":
        model, transforms = make_effnet_B0()
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, 5)
        )

    elif model_name == "vgg16":
        model, transforms = make_vgg16()
        model.classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 5)
        )

    elif model_name == "mobilenet_v3_small":
        model, transforms = make_mobilenet3_small()
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(576, 5)
        )

    else:
        raise ValueError("Unknown model")

    model = model.to(device)
    return model, transforms    

def train_all(model_names: list, 
              weight_decays: list,
              dropouts: list, 
              train_path: str, 
              test_path: str,
              save_dir: str,
              lr: float = 1e-4, 
              num_workers: int = 4, 
              batch_size: int = 128,
              epochs: int =10):
    
    # device agnostic
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Generate the output directory and folders if its not there already
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    
    try:
        for model_name in model_names:
            all_results = []
            all_history = []
            transforms = get_transforms(model_name)
            train_loader, test_loader = generate_dataloaders(
                train_path=train_path,
                test_path=test_path,
                train_transform=transforms,
                test_transform=transforms,
                batch_size=batch_size,
                num_workers=num_workers
            )
            for dropout in dropouts:
                for wd in weight_decays:
                    print("=" * 90)
                    print(f"Running: {model_name} | dropout={dropout} | weight_decay={wd}")

                    model, transforms = build_model(model_name, dropout, device)
                    # aug_transform = augmented_transform(
                    #     mean=transforms.mean,
                    #     std=transforms.std
                    # )

                    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
                    loss_fn = nn.CrossEntropyLoss()
                    print(next(model.parameters()).device)
                    history_df = main_trainer(
                        model=model,
                        train_dataloader=train_loader,
                        test_dataloader=test_loader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        total_epochs=epochs,
                        device=device
                    )
                    history_df['model_name'] = model_name
                    history_df['dropout'] = dropout
                    history_df['wd'] = wd
                    history_df.reset_index(inplace=True)
                    history_df.rename(columns={'index': 'epoch'}, inplace=True)
                    all_history.append(history_df)
                    weight_path = f"{save_dir}/{model_name}_drop{dropout}_wd{wd}.pth"
                    torch.save(model.state_dict(), weight_path)

                    all_results.append({
                        "model": model_name,
                        "dropout": dropout,
                        "weight_decay": wd,
                        "saved_weights": weight_path
                    })
            # Save current history and results for the model
            save_all_results(all_history, all_results, save_dir)

    except Exception as e:
        # In the event an error occurs, it will still save whatever has been done already
        save_all_results(all_history, all_results, save_dir)