#%% load packages
import math
import warnings
from typing import Dict, Literal

import dalex as dx
import pandas as pd

import delu  # Deep Learning Utilities: https://github.com/Yura52/delu
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm

#%% set up the model library
from rtdl_revisiting_models import MLP, ResNet, FTTransformer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% load data

df = pd.read_csv('ILPD.csv')
X = df.drop(columns='target')
y = df['target']


X.Gender = 1 if "Female" else 0
y = y-1

#%% split data
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
    X, y, train_size=0.8, random_state=0
)
X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(
    X_train, y_train, train_size=0.8, random_state=0
)

#%% convert pandas to torch
# note: X_train.values converts pandas to numpy

# X_train_tensor = torch.from_numpy(X_train.values).float()
y_train_tensor = torch.from_numpy(y_train.values).float()

# X_test_tensor = torch.from_numpy(X_test.values).float()
y_test_tensor = torch.from_numpy(y_test.values).float()

X_val_tensor = torch.from_numpy(X_val.values).float()
y_val_tensor = torch.from_numpy(y_val.values).float()

CONTINUOS_COLUMN_IDS = [0,2,3,4,5,6,7]
CATEGORICAL_COLUMN_IDS = [1]

def prepareContinuous(numpy_input):
    return torch.from_numpy(numpy_input[:, CONTINUOS_COLUMN_IDS]).float()

def prepareCategorical(numpy_input):
    result = torch.from_numpy(numpy_input[:, CATEGORICAL_COLUMN_IDS]).int()
    return result


X_train_cont_tensor = prepareContinuous(X_train.values)
X_train_cat_tensor = prepareCategorical(X_train.values)

X_test_cont_tensor = prepareContinuous(X_test.values)
X_test_cat_tensor = prepareCategorical(X_test.values)

X_val_cont_tensor = prepareContinuous(X_val.values)
X_val_cat_tensor = prepareCategorical(X_val.values)



#%% create a model 
delu.random.seed(0) # Set random seeds in all packages.
model = FTTransformer(
    n_cont_features=X_train_cont_tensor.shape[1],
    cat_cardinalities=[2],
    d_out=2, # binary classification
    **FTTransformer.get_default_kwargs(),
).to(device)

optimizer = model.make_default_optimizer()
loss_fn = F.cross_entropy

@torch.no_grad()
def evaluate(XCont, XCat, y) -> float:
    model.eval()
    y_pred = model.forward(x_cont=XCont, x_cat=XCat)[:, 1] # take probs for class 1
    y_pred = np.round(scipy.special.expit(y_pred))
    y_true = y.cpu().numpy()
    score = sklearn.metrics.roc_auc_score(y_true, y_pred)
    return score  # The higher -- the better.

print(f'Train score before training: {evaluate(X_train_cont_tensor, X_train_cat_tensor, y_train_tensor):.4f}')
print(f'Test score before training: {evaluate(X_test_cont_tensor, X_test_cat_tensor, y_test_tensor):.4f}')


#%% setup

n_epochs = 10000
patience = 10
batch_size = 128

epoch_size = math.ceil(X_train_cont_tensor.shape[0] / batch_size)
timer = delu.tools.Timer()
early_stopping = delu.tools.EarlyStopping(patience, mode="max")
best = {
    "val": -math.inf,
    "test": -math.inf,
    "epoch": -1,
}

#%% train
print(f"Device: {device.type.upper()}")
print("-" * 88 + "\n")
timer.run()
for epoch in range(n_epochs):
    for batch in tqdm(
        delu.iter_batches({'X': X_train_cont_tensor, 'xCat': X_train_cat_tensor,'y': y_train_tensor}, 
                          batch_size, shuffle=True),
        desc=f"Epoch {epoch}",
        total=epoch_size,
    ):
        model.train()
        optimizer.zero_grad()
        y_pred = model.forward(x_cont=batch['X'], x_cat=batch['xCat']) # tu podac x_cat
        loss = loss_fn(y_pred, batch["y"].long())
        loss.backward()
        optimizer.step()
    val_score = evaluate(X_val_cont_tensor, X_val_cat_tensor, y_val_tensor)
    test_score = evaluate(X_test_cont_tensor, X_test_cat_tensor, y_test_tensor)
    print(f"(val) {val_score:.4f} (test) {test_score:.4f} [time] {timer}")
    early_stopping.update(val_score)
    if early_stopping.should_stop():
        break
    if val_score > best["val"]:
        print("🌸 New best epoch! 🌸")
        best = {"val": val_score, "test": test_score, "epoch": epoch}
    # print()

print("\n\nResult:")
print(best)


#%% predict function
def predict_function(model, data):
    x_cont = prepareContinuous(data.values)
    x_cat = prepareCategorical(data.values)
    return model.forward(x_cont=x_cont, x_cat=x_cat)[:, 1].detach().numpy()

explainer = dx.Explainer(
    model=model,
    data=X_test,
    predict_function=predict_function
)

#%% explain
pdp = explainer.model_profile()
pdp.result
pdp.plot()