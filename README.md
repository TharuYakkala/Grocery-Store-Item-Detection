# DS8013_Project
![python](https://img.shields.io/badge/python-3.14.3%2B-blue)
## 📋 SUMMARY

Compares the performance of various vision models on a dataset of grocery items with various degrees of difficulty through added noise of other grocery itmes. Models testes are; VGG16, Efficientnet-b0, mobilenet_v3_small, and Resnet18.

Model training is device agnostic, so you can either install the cpu or CUDA version of pytroch and the training modules will decide
the device based on if you have cuda avaialble or not. The default is to use cuda if available.

### 📂 Folder Structure:
```
DS8013_Project/
├──src
│   ├── __init__.py
│   │── analysis
│      ├── __init__.py
│      └── plotter.py
│   │── torch_trainers
│      ├── torch_custum_models.py
│      ├── torch_data_prepper.py
│      ├── train_all_models.py
│      └── training_loop.py
├── .gitingore
├── LICENSE
├── README.md
├── requirements.txt
├── Results_analysis.ipynb
└── Train_notebook.ipynb
```

## 🔧 Environment Requirents
- Python >= 3.14.3
- All packages needed are in requirements.txt

Run the following to install required libraries
```python
pip install -r requirements.txt
```

The dataset is around 3.4GB, so its too big to add to github, you can download it from here:


**Models were trained using PyTorch**

Please install the required CUDA, and Pytorch versions by following instructions from PyTorch.
https://pytorch.org/get-started/locally/


## 📁 Detailed File Information

### SRC (Python source files)

#### Analysis
plotter.py 
- Visualization scripts that creates the comparisons of each model agaisnt various hyperparameters of dropout and weight decay.

#### 🤖 torch_trainers
torch_custom_models.py
- Contains the 3 models that freezes all weights and adds a classification head. Only the classification head is trained.

torch_data_prepper.py
- Dataset loading, and dataloader generation

train_all_models.py
- Function that trains all models that utilizes the training loop.

training_loop.py
- The core training engine that contaisn the training step, testing step, and the combined full training loop.

### 🌄 viz
Output graphs of visualizations
 - contains the comparisons of the various effects of the hyperparameters on model accuracy and loss.

### 📓 Results_analysis.ipynb
- Jupyter notebook that runs the analysis scrips for easy execution.

### 📓 Train_notebook.ipynb
- Jupyter notebook that runs the execution of all the model training.

***WARNING: MODEL TRAINING TAKES A LONG TIME***

It took us around 6 hours fully train all the models for 10 epochs each.


