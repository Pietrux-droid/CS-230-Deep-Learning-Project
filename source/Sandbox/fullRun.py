
import os
import pickle
import random
import math
import numpy as np
import pandas as pd
import matplotlib
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
os.chdir(r"C:\Users\gotta\OneDrive\Documents\Bureau\X\4A\US\Stanford\Classes\CS 230\Project\CS-230-Deep-Learning-Project")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("USING DEVICE:", device)

def save_object_to_file(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)

def read_object_from_file(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# --- Data preparation (from Rolling window TF.ipynb) ---
DATA_CSV = os.path.join("data", "FR_LMP_Forecasts", "FR_lmp_processed.csv")

def split_data(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    train = df[(df['date'] >= '2019-10-01') & (df['date'] <= '2023-09-30')].reset_index(drop=True)
    dev   = df[(df['date'] >= '2023-10-01') & (df['date'] <= '2024-09-30')].reset_index(drop=True)
    test  = df[(df['date'] >= '2024-10-01') & (df['date'] <= '2025-09-30')].reset_index(drop=True)
    return train, dev, test

def create_sliding_window(data_df, window_size, forecast_horizon=24):
    X, y = [], []
    series = data_df['EUR/MWh'].values
    for i in range(len(series) - window_size - forecast_horizon + 1):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size:i+window_size+forecast_horizon])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)

# --- Model (MLP) similar structure to Rolling window notebook ---
class MLP(nn.Module):
    def __init__(self, input_dim, layer_dims, output_dim, dropout=0.0, use_batchnorm=False, leaky_alpha=0.0):
        super().__init__()
        layers = []
        last = input_dim
        for i, d in enumerate(layer_dims):
            layers.append(nn.Linear(last, d))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(d, eps=1e-3, momentum=0.99))
            if leaky_alpha and leaky_alpha > 0:
                layers.append(nn.LeakyReLU(leaky_alpha))
            else:
                layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = d
        layers.append(nn.Linear(last, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# MAE loss (matches TF mae)
def mae_loss_fn(pred, target):
    return torch.mean(torch.abs(pred - target))

# evaluate model: predict and compute MAE and rMAE vs naive 7-day baseline
def evaluate_model(model, X_np, y_np, shift=24*7, batch_size=128, device='cpu'):
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_np), batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
    predictions = np.vstack(preds)
    if len(y_np) <= shift:
        raise ValueError("Not enough samples to compute naive baseline (need > shift samples).")
    naive_baseline = np.empty_like(y_np, dtype=float)
    naive_baseline[:shift, :] = np.nan
    naive_baseline[shift:, :] = y_np[:-shift, :]
    preds_trim = predictions[shift:]
    true = y_np[shift:]
    naive = naive_baseline[shift:]
    mae_model = np.mean(np.abs(preds_trim - true))
    mae_naive = np.mean(np.abs(naive - true))
    if mae_naive == 0:
        rmae = np.inf
        accuracy = -np.inf
    else:
        rmae = mae_model / mae_naive
        accuracy = 1.0 - rmae
    print(f"MAE model: {mae_model:.6f}, MAE naive: {mae_naive:.6f}")
    print(f"Relative MAE (rMAE): {rmae:.4f}, Accuracy (1-rMAE): {accuracy:.4f}")
    return {"predictions": preds_trim, "mae_model": mae_model, "mae_naive": mae_naive, "rmae": rmae, "accuracy": accuracy}

# --- Training routine adapted to fullRun.py style, using MAE and optional L1 ---
def run_full_train(
        X_train_np, y_train_np, X_dev_np, y_dev_np,
        layer_dims,
        dropout,
        use_batchnorm,
        leaky_alpha,
        learning_rate,
        weight_decay,
        l1_lambda,
        batch_size,
        train_amt,
        epochs,
        verbose_freq = 1
    ):
    input_dim = X_train_np.shape[1]
    output_dim = y_train_np.shape[1]
    MODEL = MLP(input_dim=input_dim, layer_dims=layer_dims, output_dim=output_dim,
                dropout=dropout, use_batchnorm=use_batchnorm, leaky_alpha=leaky_alpha).to(device)
    optimizer = torch.optim.Adam(MODEL.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-6)

    print("Model:")
    print(MODEL)
    print("Optimizer:", optimizer)
    print("Batch size:", batch_size, "Epochs:", epochs)

    train_hist = []
    val_hist = []

    X_train_use = X_train_np[:train_amt]
    y_train_use = y_train_np[:train_amt]

    X_dev_t = torch.from_numpy(X_dev_np).float().to(device)
    y_dev_t = torch.from_numpy(y_dev_np).float().to(device)

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train_use))
        X_shuf = X_train_use[perm]
        y_shuf = y_train_use[perm]

        MODEL.train()
        epoch_train_loss = 0.0
        n_seen = 0
        for i in range(0, len(X_shuf), batch_size):
            xb = torch.from_numpy(X_shuf[i:i+batch_size]).float().to(device)
            yb = torch.from_numpy(y_shuf[i:i+batch_size]).float().to(device)

            optimizer.zero_grad()
            pred = MODEL(xb)
            loss = mae_loss_fn(pred, yb)
            if l1_lambda and l1_lambda > 0:
                l1 = torch.tensor(0., device=device)
                for p in MODEL.parameters():
                    l1 = l1 + torch.sum(torch.abs(p))
                loss = loss + l1_lambda * l1
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            epoch_train_loss += loss.item() * bs
            n_seen += bs

        epoch_train_loss /= n_seen
        train_hist.append(epoch_train_loss)

        # validation
        MODEL.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val = 0
            for i in range(0, len(X_dev_np), batch_size):
                xb = torch.from_numpy(X_dev_np[i:i+batch_size]).float().to(device)
                yb = torch.from_numpy(y_dev_np[i:i+batch_size]).float().to(device)
                pred = MODEL(xb)
                loss = mae_loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)
                n_val += xb.size(0)
            val_loss /= n_val
        val_hist.append(val_loss)
        scheduler.step(val_loss)

        if verbose_freq is not None and epoch % verbose_freq == 0:
            print(f"Epoch {epoch+1}/{epochs} train_loss={epoch_train_loss:.6f} val_loss={val_loss:.6f}")

    return MODEL, train_hist, val_hist

# --- Main: load CSV, prepare sliding windows, normalize, quick experiments ---
SEED = 42
seed_everything(SEED)

df = pd.read_csv(DATA_CSV)
train_df, dev_df, test_df = split_data(df)

# normalize by train mean/std (same as TF notebook)
mean = train_df['EUR/MWh'].mean()
std  = train_df['EUR/MWh'].std()
train_df['EUR/MWh'] = (train_df['EUR/MWh'] - mean) / std
dev_df['EUR/MWh']   = (dev_df['EUR/MWh'] - mean) / std
test_df['EUR/MWh']  = (test_df['EUR/MWh'] - mean) / std

t = 366
forecast_horizon = 24
X_train, y_train = create_sliding_window(train_df, t, forecast_horizon)
X_dev,   y_dev   = create_sliding_window(dev_df, t, forecast_horizon)
X_test,  y_test  = create_sliding_window(test_df, t, forecast_horizon)

# experiment config(s) (keeps logic/content from notebook)
all_exp_params = {
    "default": {
        "layer_dims": [256, 128],
        "dropout": 0.1,
        "use_batchnorm": True,
        "leaky_alpha": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "l1_lambda": 0.0,
        "batch_size": 128,
        "train_amt": len(X_train),   # use all by default
        "epochs": 50
    },
    "quick_small": {
        "layer_dims": [512, 256, 128],
        "dropout": 0.2,
        "use_batchnorm": True,
        "leaky_alpha": 0.01,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "l1_lambda": 0.0,
        "batch_size": 128,
        "train_amt": max(500, int(len(X_train)*0.25)),
        "epochs": 10
    }
}

results_save_path = "rolling_window_exp_results.pkl"
plot_suffix = "_loss_plot.jpg"
exp_results = {}

for tag, params in all_exp_params.items():
    print("Running experiment:", tag)
    model, train_hist, val_hist = run_full_train(
        X_train, y_train, X_dev, y_dev,
        layer_dims = params["layer_dims"],
        dropout = params["dropout"],
        use_batchnorm = params["use_batchnorm"],
        leaky_alpha = params["leaky_alpha"],
        learning_rate = params["learning_rate"],
        weight_decay = params["weight_decay"],
        l1_lambda = params.get("l1_lambda", 0.0),
        batch_size = params["batch_size"],
        train_amt = params["train_amt"],
        epochs = params["epochs"],
        verbose_freq = 1
    )
    # evaluate dev set (rMAE etc)
    eval_res = evaluate_model(model, X_dev, y_dev, device=device)
    exp_results[tag] = {"train_loss": train_hist, "val_loss": val_hist, "eval": eval_res}

save_object_to_file(exp_results, results_save_path)
exp_results = read_object_from_file(results_save_path)

for tag, r in exp_results.items():
    tr = r["train_loss"]
    va = r["val_loss"]
    plt.clf()
    plt.plot(range(len(tr)), tr, label="Train Loss")
    plt.plot(range(len(va)), va, label="Validation Loss")
    plt.title(f"{tag} Loss vs Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend(loc="upper right")
    plt.savefig(tag + plot_suffix)
    plt.clf()
