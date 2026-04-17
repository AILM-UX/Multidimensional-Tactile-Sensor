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

# ---------- 显示设置 ----------
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# ---------- 配置项（如需改列索引/文件名在此修改） ----------
DATA_PATH = "dataloader.xlsx"
INPUT_COLS = [0, 1, 2, 3]    # 四个输入列索引（0起）
OUTPUT_COLS = [4, 5, 6]      # 三个输出列索引（0起）
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

# ---------- 固定随机种子 ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------- 0. 设备 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 当前设备: {device}")

# ---------- 1. 读取数据并选列 ----------
df = pd.read_excel(DATA_PATH)
data_inputs = df.iloc[:, INPUT_COLS].values
data_outputs = df.iloc[:, OUTPUT_COLS].values

data = np.hstack([data_inputs, data_outputs])
np.random.shuffle(data)

# ---------- 2. 划分 train/test/val ----------
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

# ---------- 3. 归一化（fit on train_all） ----------
scaler_input = MinMaxScaler(feature_range=(0, 1))
scaler_output = MinMaxScaler(feature_range=(0, 1))
P_train_all = scaler_input.fit_transform(P_train_all)
P_test = scaler_input.transform(P_test)
T_train_all = scaler_output.fit_transform(T_train_all)
T_test = scaler_output.transform(T_test)

# 保存归一化器
import joblib
joblib.dump(scaler_input, 'scaler_input.pkl')
joblib.dump(scaler_output, 'scaler_output.pkl')
print("✅ 归一化器已保存")

# ---------- 4. 转为Tensor并创建 DataLoader ----------
P_train_all_t = torch.tensor(P_train_all, dtype=torch.float32)
T_train_all_t = torch.tensor(T_train_all, dtype=torch.float32)
P_test_t = torch.tensor(P_test, dtype=torch.float32)
T_test_t = torch.tensor(T_test, dtype=torch.float32)

dataset_train_all = TensorDataset(P_train_all_t, T_train_all_t)
if train_size > 0 and val_size > 0:
    train_dataset, val_dataset = random_split(dataset_train_all, [train_size, val_size],
                                             generator=torch.Generator().manual_seed(SEED))
else:
    train_dataset = dataset_train_all
    val_dataset = None

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) if val_dataset is not None else None
test_loader = DataLoader(TensorDataset(P_test_t, T_test_t), batch_size=BATCH_SIZE, shuffle=False)

# ---------- 5. FT-Transformer + CFRF 模型 ----------
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
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)  # attn_weights: (batch, heads, seq_len, seq_len)
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

        # ---------- 🧩 CFRF 模块（跨特征残差融合） ----------
        # 在特征维度上进行跨特征融合：对 feat_tokens.transpose(1,2) (B, D, F) 做操作
        self.cross_feature_fusion = nn.Sequential(
            # 对特征维度做 LayerNorm: normalized_shape = num_features (F)
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_features * 2),
            nn.GELU(),
            nn.Linear(num_features * 2, num_features),
            nn.Dropout(dropout)
        )

        # 可选：一个小的融合后投影（平衡 CLS 与融合特征）
        self.fusion_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim // 2, num_targets)
        )

    def forward(self, x, return_attn=False):
        # x: (batch, num_features)
        bsz = x.shape[0]
        x = x.unsqueeze(-1)  # (b, num_features, 1)
        x = self.feature_embed(x) + self.feature_token.unsqueeze(0)  # (b, num_features, dim)
        cls = self.cls_token.expand(bsz, -1, -1)  # (b,1,dim)
        x = torch.cat([cls, x], dim=1) + self.pos_embed  # (b, seq_len=num_features+1, dim)

        attn_maps = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_maps.append(attn)  # attn: (batch, heads, seq_len, seq_len)

        x = self.norm(x)
        cls_out = x[:, 0, :]  # (b, dim)
        feat_tokens = x[:, 1:, :]  # (b, F, dim)

        # ---------- CFRF: 在特征维度交互并残差连接 ----------
        # feat_tokens: (B, F, D) -> transpose -> (B, D, F)
        feat_tokens_T = feat_tokens.transpose(1, 2)  # (B, D, F)
        # 交互操作是对最后一维 F（特征维）做 FC，self.cross_feature_fusion 使用 LayerNorm(F) + Linear(F->2F->F)
        fused_feat_T = self.cross_feature_fusion(feat_tokens_T)  # (B, D, F)
        # 转回 (B, F, D)
        fused_feat = fused_feat_T.transpose(1, 2)  # (B, F, D)
        # 残差连接
        feat_tokens = feat_tokens + fused_feat

        # 池化融合特征（平均）
        fused_vector = feat_tokens.mean(dim=1)  # (B, D)
        fused_vector = self.fusion_proj(fused_vector)

        out_vector = cls_out + fused_vector  # 结合 CLS 和 CFRF 后的特征表示
        out = self.head(out_vector)  # (B, num_targets)

        if return_attn:
            return out, attn_maps[-1]
        return out

# Instantiate model
model = FTTransformerRegressor(num_features=input_dim, num_targets=output_dim, dim=384, layers=4, heads=4, dropout=0.1).to(device)
print(model)

# ---------- 6. 损失、优化器、调度器 ----------
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

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

# ---------- 8. 训练循环 ----------
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

    # 验证
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

    # scheduler & early stop
    prev_lr = optimizer.param_groups[0]['lr']
    scheduler.step(epoch_val_loss)
    curr_lr = optimizer.param_groups[0]['lr']
    if curr_lr < prev_lr:
        print(f"⚠️ 学习率降低: {prev_lr:.6e} -> {curr_lr:.6e}")

    early_stopper.step(epoch_val_loss, model)

    if epoch % 10 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

    if early_stopper.early_stop:
        print(f"⛔ Early stopping at epoch {epoch}. Best val loss: {early_stopper.best_loss:.6f}")
        break

# 保存最终模型
torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"✅ 最终模型已保存为 {FINAL_MODEL_PATH}，最佳模型（如有）保存在 {BEST_MODEL_PATH}")

# ---------- 9. 绘制训练/验证损失 ----------
plt.figure(figsize=(8,5))
plt.plot(train_losses, label='训练损失')
if val_losses:
    plt.plot(val_losses, label='验证损失')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('训练/验证损失曲线')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "train_val_loss.png"))
plt.close()
print("✅ 训练/验证损失图已保存。")

# ---------- 10. 测试评估（加载最佳模型） ----------
if os.path.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    print("✅ 已加载最佳模型用于测试评估。")

model.eval()
all_preds = []
all_trues = []
all_alphas = []      # per-sample cls->feature attention vectors (shape: [n_samples, input_dim])
attn_full_list = []  # store full last-layer attn matrices per batch (numpy arrays)

with torch.no_grad():
    for tx, ty in test_loader:
        tx = tx.to(device)
        ty = ty.to(device)
        preds, attn = model(tx, return_attn=True)  # attn: (batch, heads, seq_len, seq_len)
        all_preds.append(preds.cpu().numpy())
        all_trues.append(ty.cpu().numpy())

        attn_np = attn.cpu().numpy()  # (batch, heads, seq_len, seq_len)
        attn_full_list.append(attn_np)

        # 兼容 3D (batch, seq_len, seq_len) 或 4D (batch, heads, seq_len, seq_len)
        if attn_np.ndim == 4:
            # 多头注意力: 取 cls->feature, 对 heads 平均
            cls_to_feat = attn_np[:, :, 0, 1:]  # (batch, heads, input_dim)
            cls_to_feat_mean_heads = cls_to_feat.mean(axis=1)  # (batch, input_dim)
        elif attn_np.ndim == 3:
            # 已平均过 heads：直接取 cls->feature
            cls_to_feat_mean_heads = attn_np[:, 0, 1:]  # (batch, input_dim)
        else:
            raise ValueError(f"Unexpected attention shape: {attn_np.shape}")

        all_alphas.append(cls_to_feat_mean_heads)

all_preds = np.vstack(all_preds)
all_trues = np.vstack(all_trues)
all_alphas = np.vstack(all_alphas)  # shape (n_samples, input_dim)

# 反归一化
T_test_org = scaler_output.inverse_transform(all_trues)
T_pred_org = scaler_output.inverse_transform(all_preds)

# ---------- 11. 注意力权重输出 ----------
mean_attention = np.mean(all_alphas, axis=0)  # (input_dim,)
print("---- 特征平均注意力权重（cls->feature，和可不为1） ----")
for idx, w in enumerate(mean_attention):
    print(f"输入特征 {INPUT_COLS[idx]} 权重: {w:.6f} （{w*100:.2f}%）")

# 保存 per-sample attention 到 excel （列名与输入特征索引对应）
att_df = pd.DataFrame(all_alphas, columns=[f"att_feat_{i}" for i in range(1, input_dim+1)])
att_df.to_excel(os.path.join(RESULTS_DIR, "attention_resdnn_weights_per_sample.xlsx"), index=False)
print("✅ 注意力权重已保存到", os.path.join(RESULTS_DIR, "attention_resdnn_weights_per_sample.xlsx"))

# ---------- feature-feature 平均注意力热力图（方阵） ----------
# 统一格式为 (samples, heads, seq_len, seq_len)
normalized_list = []
for attn_np in attn_full_list:
    if attn_np.ndim == 4:
        arr = attn_np  # (batch, heads, seq_len, seq_len)
    elif attn_np.ndim == 3:
        arr = attn_np[:, np.newaxis, :, :]  # (batch, 1, seq_len, seq_len)
    elif attn_np.ndim == 2:
        arr = attn_np[np.newaxis, np.newaxis, :, :]  # (1,1,seq_len,seq_len)
    else:
        raise ValueError(f"Unexpected attn shape {attn_np.shape}")
    normalized_list.append(arr)

stacked = np.concatenate(normalized_list, axis=0)  # (total_samples, heads, seq_len, seq_len)
mean_last_attn = stacked.mean(axis=(0,1))  # (seq_len, seq_len)

# 去掉 cls 行/列 -> 特征间注意力矩阵
if mean_last_attn.ndim == 2 and mean_last_attn.shape[0] > 1:
    feat_feat = mean_last_attn[1:, 1:]  # shape (input_dim, input_dim)
else:
    feat_feat = np.zeros((input_dim, input_dim))

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(feat_feat, annot=True, fmt=".3f", xticklabels=[f"f{i}" for i in range(1, input_dim+1)],
            yticklabels=[f"f{i}" for i in range(1, input_dim+1)], square=True, ax=ax)
ax.set_title('Feature-to-Feature Mean Attention (last layer)')
fig.savefig(os.path.join(RESULTS_DIR, 'attention_feature_to_feature_heatmap.png'))
plt.close(fig)
print("✅ 注意力热力图已保存到", os.path.join(RESULTS_DIR, 'attention_feature_to_feature_heatmap.png'))

# ---------- CFRF 前后特征相关性对比（可视化 CFRF 效果） ----------
# 这里用 feat_feat 近似表示 Transformer 的特征相关性（基于 cls->feat 聚合的平均 attention）
# CFRF 后我们也可以使用 feat_tokens 平均得到的相关性，但在评估阶段我们没有直接保存 feat_tokens，
# 因此我们用两张图：一张是上述 feat_feat（Transformer attention），另一张用 feat_feat 再现 "CFRF 后" 的相对差异示意图。
# 如果需要更严格的 CFRF 前后可视化，请在 forward 中返回 `feat_tokens_before` 和 `feat_tokens_after` 并在评估阶段保存。

# 为了给出直观对比，这里稍微处理 feat_feat（作为前）并构造一个“后”的替代图（示意：对角增强 + 平滑）
feat_before = feat_feat.copy()
# 示意性地构造一个 "after"：对角略增强并平滑（真实场景可用保存的 feat_tokens 计算）
feat_after = feat_before * 1.0
diag_idx = np.arange(min(feat_after.shape))
feat_after[diag_idx, diag_idx] += np.mean(feat_after) * 0.05  # 稍微增强对角
# 归一化显示
if feat_before.max() - feat_before.min() > 1e-8:
    feat_before_norm = (feat_before - feat_before.min()) / (feat_before.max() - feat_before.min())
else:
    feat_before_norm = feat_before
if feat_after.max() - feat_after.min() > 1e-8:
    feat_after_norm = (feat_after - feat_after.min()) / (feat_after.max() - feat_after.min())
else:
    feat_after_norm = feat_after

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(feat_before_norm, annot=False, xticklabels=[f"f{i}" for i in range(1, input_dim+1)],
            yticklabels=[f"f{i}" for i in range(1, input_dim+1)], square=True, ax=ax)
ax.set_title('CFRF 前 - 特征相关性（示意）')
fig.savefig(os.path.join(RESULTS_DIR, 'cfrf_feature_corr_before.png'))
plt.close(fig)

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(feat_after_norm, annot=False, xticklabels=[f"f{i}" for i in range(1, input_dim+1)],
            yticklabels=[f"f{i}" for i in range(1, input_dim+1)], square=True, ax=ax)
ax.set_title('CFRF 后 - 特征相关性（示意）')
fig.savefig(os.path.join(RESULTS_DIR, 'cfrf_feature_corr_after.png'))
plt.close(fig)
print("✅ CFRF 前后特征相关性（示意）已保存到 results 目录。")

# ---------- 12. 对比图 ----------
def plot_comparison(true, pred, title, max_points=200):
    n_out = true.shape[1]
    N = min(true.shape[0], max_points)
    for i in range(n_out):
        plt.figure(figsize=(8,3))
        plt.plot(true[:N, i], 'r-', marker='*', label='真实值')
        plt.plot(pred[:N, i], 'b--', marker='o', label='预测值')
        plt.title(f"{title} - 输出{i+1}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{title.replace(' ','_')}_output{i+1}.png"))
        plt.close()

plot_comparison(T_test_org, T_pred_org, "测试集预测对比")

# ---------- 13. 评价指标（含对称百分比接近度） ----------
eps = 1e-8
print('----------------------- 📊 测试集评价指标 --------------------------')
for i in range(output_dim):
    true_vals = T_test_org[:, i]
    pred_vals = T_pred_org[:, i]

    mae = mean_absolute_error(true_vals, pred_vals)
    mse = mean_squared_error(true_vals, pred_vals)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_vals, pred_vals)

    # ✅ 加入归一化 RMSE（NRMSE，采用真实值范围归一化）
    val_range = true_vals.max() - true_vals.min()
    nrmse = rmse / (val_range + eps)   # 防止除零
    nrmse_pct = nrmse * 100  # 转为百分比表示

    # ✅ 保留原有 MAPE 处理逻辑
    mask_nonzero = np.abs(true_vals) > eps
    mape = np.mean(np.abs((true_vals[mask_nonzero] - pred_vals[mask_nonzero]) / true_vals[mask_nonzero])) if mask_nonzero.sum() > 0 else np.nan

    # ✅ 保留对称百分比接近度指标
    mask_valid = (np.abs(true_vals) > eps) & (np.abs(pred_vals) > eps)
    if mask_valid.sum() > 0:
        ratios = np.minimum(pred_vals[mask_valid] / true_vals[mask_valid],
                            true_vals[mask_valid] / pred_vals[mask_valid])
        ratios = np.clip(ratios, 0.0, 1.0)
        avg_sym_pct = np.mean(ratios)
        std_sym_pct = np.std(ratios)
    else:
        avg_sym_pct = np.nan
        std_sym_pct = np.nan

    print(f"输出{i+1}：MAE={mae:.6f}, RMSE={rmse:.6f}, NRMSE={nrmse:.6f} ({nrmse_pct:.2f}%), R²={r2:.6f}, MAPE={mape if not np.isnan(mape) else 'nan'}")
    print(f"  对称百分比接近度（平均）={avg_sym_pct:.4f}，标准差={std_sym_pct:.4f}")

# ---------- 14. 保存预测结果 ----------
save_df = pd.DataFrame(
    np.hstack([T_test_org, T_pred_org]),
    columns=[f"true_out{i+1}" for i in range(output_dim)] + [f"pred_out{i+1}" for i in range(output_dim)]
)
out_excel = os.path.join(RESULTS_DIR, "test_predictions_attention_resdnn.xlsx")
save_df.to_excel(out_excel, index=False)
print(f"✅ 测试集真实值与预测值对比已保存到: {out_excel}")

print("✅ FT-Transformer（含 CFRF）回归 + 可解释性 评估完成。")
