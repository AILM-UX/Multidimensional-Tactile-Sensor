import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ---------- Display Settings ----------
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- Configuration ----------
DATA_PATH = "dataloader.xlsx"
INPUT_COLS = [0, 1, 2, 3]      # Four input column indices (starting from 0)
OUTPUT_COLS = [4, 5, 6]        # Three output column indices (starting from 0)
TRAIN_RATIO = 0.7
VAL_RATIO_WITHIN_TRAIN = 0.15
BATCH_SIZE = 64
LR = 0.0003
WEIGHT_DECAY = 1e-4
EPOCHS = 500
PATIENCE = 40
CLIP_NORM = 1.0
SEED = 42
BEST_MODEL_PATH = "ft_best.pth"
FINAL_MODEL_PATH = "ft_final.pth"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- Fix Random Seed ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------- 0. Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

# ---------- 1. Read Data and Select Columns ----------
df = pd.read_excel(DATA_PATH)
data_inputs = df.iloc[:, INPUT_COLS].values
data_outputs = df.iloc[:, OUTPUT_COLS].values

data = np.hstack([data_inputs, data_outputs])
np.random.shuffle(data)

# ---------- 2. Split train/test/val ----------
num_samples = data.shape[0]
num_train = int(round(TRAIN_RATIO * num_samples))
train_all = data[:num_train]
test = data[num_train:]

val_size = int(round(VAL_RATIO_WITHIN_TRAIN * train_all.shape[0]))
train_size = train_all.shape[0] - val_size

input_dim = len(INPUT_COLS)
output_dim = len(OUTPUT_COLS)

P_train_all = train_all[:, :input_dim]
T_train_all = train_all[:, input_dim: input_dim + output_dim]
P_test = test[:, :input_dim]
T_test = test[:, input_dim: input_dim + output_dim]

# ---------- 3. Normalization (fit on train_all) ----------
scaler_input = MinMaxScaler(feature_range=(0, 1))
scaler_output = MinMaxScaler(feature_range=(0, 1))
P_train_all = scaler_input.fit_transform(P_train_all)
P_test = scaler_input.transform(P_test)
T_train_all = scaler_output.fit_transform(T_train_all)
T_test = scaler_output.transform(T_test)

# Save scalers
joblib.dump(scaler_input, 'scaler_input.pkl')
joblib.dump(scaler_output, 'scaler_output.pkl')
print(" Scalers have been saved")

# ---------- 4. Convert to Tensor and Create DataLoader ----------
P_train_all_t = torch.tensor(P_train_all, dtype=torch.float32)
T_train_all_t = torch.tensor(T_train_all, dtype=torch.float32)
P_test_t = torch.tensor(P_test, dtype=torch.float32)
T_test_t = torch.tensor(T_test, dtype=torch.float32)

dataset_train_all = TensorDataset(P_train_all_t, T_train_all_t)
if train_size > 0 and val_size > 0:
    train_dataset, val_dataset = random_split(
        dataset_train_all,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
else:
    train_dataset = dataset_train_all
    val_dataset = None

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset is not None else None
test_loader = DataLoader(TensorDataset(P_test_t, T_test_t), batch_size=BATCH_SIZE, shuffle=False)

# ---------- 5. FT-Transformer + CFRF Model ----------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x, attn_weights


class FTTransformerRegressor(nn.Module):
    def __init__(self, num_features, num_targets=3, dim=128, layers=3, heads=4, dropout=0.2):
        super().__init__()
        self.num_features = num_features
        self.feature_embed = nn.Linear(1, dim)
        self.feature_token = nn.Parameter(torch.randn(num_features, dim))  # per-feature bias/token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_features + 1, dim))
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, 4, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

        # ---------- CFRF Module (Cross-Feature Residual Fusion) ----------
        # Perform cross-feature fusion on feat_tokens.transpose(1, 2), shape: (B, D, F)
        self.cross_feature_fusion = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_features * 2),
            nn.GELU(),
            nn.Linear(num_features * 2, num_features),
            nn.Dropout(dropout)
        )

        # Optional projection after fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_targets)
        )

    def forward(self, x, return_attn=False):
        # x: (batch, num_features)
        bsz = x.shape[0]
        x = x.unsqueeze(-1)  # (b, num_features, 1)
        x = self.feature_embed(x) + self.feature_token.unsqueeze(0)  # (b, num_features, dim)
        cls = self.cls_token.expand(bsz, -1, -1)  # (b, 1, dim)
        x = torch.cat([cls, x], dim=1) + self.pos_embed  # (b, seq_len=num_features+1, dim)

        attn_maps = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_maps.append(attn)

        x = self.norm(x)
        cls_out = x[:, 0, :]        # (b, dim)
        feat_tokens = x[:, 1:, :]   # (b, F, dim)

        # ---------- CFRF: Cross-feature interaction and residual connection ----------
        feat_tokens_T = feat_tokens.transpose(1, 2)   # (B, D, F)
        fused_feat_T = self.cross_feature_fusion(feat_tokens_T)  # (B, D, F)
        fused_feat = fused_feat_T.transpose(1, 2)     # (B, F, D)
        feat_tokens = feat_tokens + fused_feat

        # Pool fused features (mean)
        fused_vector = feat_tokens.mean(dim=1)  # (B, D)
        fused_vector = self.fusion_proj(fused_vector)

        out_vector = cls_out + fused_vector
        out = self.head(out_vector)  # (B, num_targets)

        if return_attn:
            return out, attn_maps[-1]
        return out


# Instantiate model
model = FTTransformerRegressor(
    num_features=input_dim,
    num_targets=output_dim,
    dim=384,
    layers=4,
    heads=4,
    dropout=0.1
).to(device)
print(model)

# ---------- 6. Loss Function, Optimizer, Scheduler ----------
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# ---------- 7. EarlyStopping ----------
class EarlyStopping:
    def __init__(self, patience=PATIENCE, delta=1e-6, save_path=BEST_MODEL_PATH):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.save_path = save_path

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


early_stopper = EarlyStopping()

# ---------- 8. Training Loop ----------
train_losses = []
val_losses = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        running_loss += loss.item() * batch_x.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    # Validation
    if val_loader is not None:
        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx = vx.to(device)
                vy = vy.to(device)
                vpreds = model(vx)
                vloss = criterion(vpreds, vy)
                val_running += vloss.item() * vx.size(0)

        epoch_val_loss = val_running / len(val_loader.dataset) if len(val_loader.dataset) > 0 else float('inf')
        val_losses.append(epoch_val_loss)
    else:
        epoch_val_loss = epoch_train_loss

    # Scheduler & early stop
    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(epoch_val_loss)
    curr_lr = optimizer.param_groups[0]['lr']

    if curr_lr < prev_lr:
        print(f" Learning rate reduced: {prev_lr:.6e} -> {curr_lr:.6e}")

    early_stopper.step(epoch_val_loss, model)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    if early_stopper.early_stop:
        print(f" Early stopping at epoch {epoch}. Best val loss: {early_stopper.best_loss:.6f}")
        break

# Save final model
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f" Final model saved as {FINAL_MODEL_PATH}, best model saved at {BEST_MODEL_PATH}")

# ---------- 9. Plot Training/Validation Loss ----------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
if val_losses:
    plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "train_val_loss.png"))
plt.close()
print(" Training/validation loss figure has been saved.")

# ---------- 10. Test Evaluation (Load Best Model) ----------
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print(" Best model loaded for test evaluation.")

model.eval()
all_preds = []
all_trues = []
all_alphas = []      # per-sample cls->feature attention vectors
attn_full_list = []  # full last-layer attention matrices per batch

with torch.no_grad():
    for tx, ty in test_loader:
        tx = tx.to(device)
        ty = ty.to(device)
        preds, attn = model(tx, return_attn=True)
        all_preds.append(preds.cpu().numpy())
        all_trues.append(ty.cpu().numpy())

        attn_np = attn.cpu().numpy()
        attn_full_list.append(attn_np)

        # Compatible with 3D or 4D attention
        if attn_np.ndim == 4:
            cls_to_feat = attn_np[:, :, 0, 1:]             # (batch, heads, input_dim)
            cls_to_feat_mean_heads = cls_to_feat.mean(axis=1)
        elif attn_np.ndim == 3:
            cls_to_feat_mean_heads = attn_np[:, 0, 1:]
        else:
            raise ValueError(f"Unexpected attention shape: {attn_np.shape}")

        all_alphas.append(cls_to_feat_mean_heads)

all_preds = np.vstack(all_preds)
all_trues = np.vstack(all_trues)
all_alphas = np.vstack(all_alphas)

# Inverse normalization
T_test_org = scaler_output.inverse_transform(all_trues)
T_pred_org = scaler_output.inverse_transform(all_preds)

# ---------- 11. Comparison Plots ----------
def plot_comparison(true, pred, title, max_points=200):
    n_out = true.shape[1]
    N = min(true.shape[0], max_points)
    for i in range(n_out):
        plt.figure(figsize=(8,3))
        plt.plot(true[:N, i], 'r-', marker='*', label='actual')
        plt.plot(pred[:N, i], 'b--', marker='o', label='predict')
        plt.title(f"{title} - force{i+1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{title.replace(' ','_')}_output{i+1}.png"))
        plt.close()

plot_comparison(T_test_org, T_pred_org, "Comparison of Predictions on the Test Set")

# ---------- 12. Evaluation Metrics (Including Symmetric Percentage Closeness) ----------
eps = 1e-8
print('----------------------- Test Set Evaluation Metrics --------------------------')
for i in range(output_dim):
    true_vals = T_test_org[:, i]
    pred_vals = T_pred_org[:, i]

    mae = mean_absolute_error(true_vals, pred_vals)
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_vals, pred_vals)

    # NRMSE normalized by the range of true values
    val_range = true_vals.max() - true_vals.min()
    nrmse = rmse / (val_range + eps)
    nrmse_pct = nrmse * 100

    # MAPE with zero masking
    mask_nonzero = np.abs(true_vals) > eps
    mape = np.mean(
        np.abs((true_vals[mask_nonzero] - pred_vals[mask_nonzero]) / true_vals[mask_nonzero])
    ) if mask_nonzero.sum() > 0 else np.nan

    print(
        f"Output {i+1}: MAE={mae:.6f}, RMSE={rmse:.6f}, "
        f"NRMSE={nrmse:.6f} ({nrmse_pct:.2f}%), R²={r2:.6f}, "
        f"MAPE={mape if not np.isnan(mape) else 'nan'}"
    )


# ---------- 13. Save Prediction Results ----------
save_df = pd.DataFrame(
    np.hstack([T_test_org, T_pred_org]),
    columns=[f"true_out{i+1}" for i in range(output_dim)] +
            [f"pred_out{i+1}" for i in range(output_dim)]
)
out_excel = os.path.join(RESULTS_DIR, "test_predictions.xlsx")
save_df.to_excel(out_excel, index=False)
print(f" Comparison between true and predicted values on the test set has been saved to: {out_excel}")

print(" FT-Transformer (with CFRF) regression and interpretability evaluation completed.")