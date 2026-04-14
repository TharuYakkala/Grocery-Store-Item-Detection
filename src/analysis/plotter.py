import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import warnings

def variance_analyzer(history_path: str):
    hist = pd.read_csv(history_path)
    hist['test_var'] = (hist["train_acc"] - hist['test_acc'])**2
    hist['loss_var'] = (hist['train_loss'] - hist['test_loss'])**2
    hist.groupby('model_name')[['test_var', 'loss_var']].mean().plot(kind='bar')
    plt.title("Test accuracy and loss variance")
    plt.ylabel("Var")
    viz_path = Path("./viz")
    plt.savefig(viz_path / "model_variances.png")
    plt.show()
    
    hist[hist['model_name']!='vgg16'].groupby('model_name')[['test_var', 'loss_var']].mean().plot(kind='bar')
    plt.title("Test accuracy and loss variance")
    plt.ylabel("Var")
    plt.savefig(viz_path / "model_variances2.png")
    plt.show()
    
def generate_model_plots(history_path: str):
    hist = pd.read_csv(history_path)
    models = hist['model_name'].unique()

    for model in models:
        df_model = hist[hist['model_name']==model]
        fig, axes = plt.subplots(1, 3, figsize=(18,5), sharey=True) 
        dropouts = sorted(df_model['dropout'].unique())
        for i, dropout in enumerate(dropouts):
            ax = axes[i]
            group_d = df_model[df_model['dropout']==dropout]
            for wd, group in group_d.groupby('wd'):
                group = group.sort_values('epoch')
                ax.plot(
                    group['epoch'],
                    group['test_loss'],
                    label=f"wd={wd}"
                )
                ax.set_title(f"{model} | Dropout = {dropout}")
                ax.set_xlabel("Epoch")
                ax.grid(True)

        axes[0].set_ylabel("Test Loss")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels))

        plt.tight_layout(rect=[0, 0, 1, 0.9])
        viz_path = Path(f"./viz/{model}")
        viz_path.mkdir(exist_ok=True, parents=True)
        graph_path = viz_path / f"{model}_hyperparam_plot.png"
        
        if not graph_path.exists():
            plt.savefig(viz_path / f"{model}_hyperparam_plot.png")
        else:
            # To avoid overidding over and over
            warnings.warn("Plot visuals exists, skipping save.")
        plt.show()
        
def best_params_per_model(history_path: str, metric: str):
    df = pd.read_csv(history_path)
    # Best hyperparams per model
    if metric in ('train_acc', 'test_acc'):
        best_params = (
            df.groupby(['model_name', 'dropout', 'wd'])
            .agg({metric: 'max'})
            .reset_index()
        )

        best_per_model = best_params.loc[
            best_params.groupby('model_name')[metric].idxmax()
        ].reset_index(drop=True)
    elif metric in ('train_loss', 'test_loss'):
        best_params = (
        df.groupby(['model_name', 'dropout', 'wd'])
        .agg({metric: 'min'})
        .reset_index()
        )

        best_per_model = best_params.loc[
            best_params.groupby('model_name')[metric].idxmin()
        ].reset_index(drop=True)
    else:
        raise ValueError("Metric must be one of train_loss, test_loss, train_acc, test_acc.")
    return best_per_model