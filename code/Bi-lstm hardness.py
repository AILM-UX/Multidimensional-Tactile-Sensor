import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Set random seed
np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# -------------------- 1. Load Dataset --------------------
def load_data_from_csv(csv_path, window_size=100, stride=50):
    """
    Load four-channel resistance sensor data from a CSV file
    and generate samples using a sliding window.
    Feature columns: CH0_R, CH1_R, CH2_R, CH3_R
    Label column: hardness (automatically converted to integer classes)
    """
    df = pd.read_csv(csv_path)

    feature_cols = ['CH0_R', 'CH1_R', 'CH2_R', 'CH3_R']
    missing_cols = [col for col in feature_cols + ['hardness'] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    X_raw = df[feature_cols].values.astype(np.float32)   # shape: (total_time_steps, 4)
    hardness_raw = df['hardness'].values

    # Convert hardness to integer classes
    # If already integers, cast directly; otherwise use factorize
    unique_hard = np.unique(hardness_raw)
    if np.all(np.mod(unique_hard, 1) == 0):
        y_raw = hardness_raw.astype(int)
    else:
        y_raw, _ = pd.factorize(hardness_raw)

    # Generate samples using sliding window
    n = len(X_raw)
    if n < window_size:
        raise ValueError(f"Data length {n} is smaller than window size {window_size}, unable to generate samples")

    X_list, y_list = [], []
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        X_window = X_raw[start:end]                     # (window_size, 4)
        y_window = y_raw[start:end]

        # Use the mode (most frequent hardness value) as the window label
        labels, counts = np.unique(y_window, return_counts=True)
        y_label = labels[np.argmax(counts)]
        X_list.append(X_window)
        y_list.append(y_label)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"Loaded data from CSV: X shape {X.shape}, y shape {y.shape}")
    print(f"Feature columns used: {feature_cols}")
    print(f"Class distribution: {np.bincount(y)}")
    return X, y


# Load data (window size and stride can be adjusted according to actual data)
csv_path = "DATA.csv"
X, y = load_data_from_csv(csv_path, window_size=100, stride=1)

# Split training set and test set (stratified sampling)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize data (along feature dimension)
scaler = StandardScaler()
nsamples, nsteps, nfeats = X_train.shape
X_train_reshaped = X_train.reshape(-1, nfeats)
scaler.fit(X_train_reshaped)
X_train_scaled = scaler.transform(X_train_reshaped).reshape(nsamples, nsteps, nfeats)
X_test_scaled = scaler.transform(X_test.reshape(-1, nfeats)).reshape(X_test.shape[0], nsteps, nfeats)

# Convert to PyTorch tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
batch_size = 32
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# -------------------- 2. Define Bi-LSTM Model (input feature size changed to 4) --------------------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

    def forward(self, x):
        device = x.device
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]

        out = self.batch_norm(out)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


# Hyperparameter configuration (input_size changed to 4)
input_size = 4
hidden_size = 128
num_layers = 2
num_classes = len(np.unique(y))
learning_rate = 0.002
num_epochs = 300
weight_decay = 1e-5


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = BiLSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)


# -------------------- 3. Train Model --------------------
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        model.eval()
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        epoch_train_acc = 100 * correct_train / total_train
        epoch_test_acc = 100 * correct_test / total_test
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)
        train_accs.append(epoch_train_acc)
        test_accs.append(epoch_test_acc)

        history.append({
            'Epoch': epoch + 1,
            'Train Loss': epoch_train_loss,
            'Test Loss': epoch_test_loss,
            'Train Accuracy (%)': epoch_train_acc,
            'Test Accuracy (%)': epoch_test_acc,
            'Learning Rate': current_lr
        })

        scheduler.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'  Learning Rate: {current_lr:.8f}')
        print(f'  Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_acc:.2f}%')
        print(f'  Test Loss: {epoch_test_loss:.4f}, Test Accuracy: {epoch_test_acc:.2f}%')
        print('-' * 50)

    history_df = pd.DataFrame(history)
    history_df.to_excel('training_history_4ch.xlsx', index=False)
    print("\nLoss and accuracy for each epoch have been saved to: training_history_4ch.xlsx")

    return train_losses, test_losses, train_accs, test_accs


print("Start training the model...")
train_losses, test_losses, train_accs, test_accs = train_model(
    model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device
)


# -------------------- 4. Visualize Training Results --------------------
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses, label='Test Loss')
ax1.set_title('Train/Test Loss Curve')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, label='Train Accuracy')
ax2.plot(test_accs, label='Test Accuracy')
ax2.set_title('Train/Test Accuracy Curve')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)


# -------------------- 5. Confusion Matrix --------------------
def generate_and_plot_confusion_matrix(model, test_loader, device, ax):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    class_names = [str(i) for i in range(len(np.unique(all_labels)))]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Normalized Confusion Matrix (Test Set)')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('True Class')

    cm_df = pd.DataFrame(cm, index=[f'True_{c}' for c in class_names],
                         columns=[f'Pred_{c}' for c in class_names])
    cm_normalized_df = pd.DataFrame(cm_normalized, index=[f'True_{c}' for c in class_names],
                                    columns=[f'Pred_{c}' for c in class_names])
    with pd.ExcelWriter('confusion_matrix_4ch.xlsx') as writer:
        cm_df.to_excel(writer, sheet_name='Raw Confusion Matrix')
        cm_normalized_df.to_excel(writer, sheet_name='Normalized Confusion Matrix')
    print("\nConfusion matrix has been saved to: confusion_matrix_4ch.xlsx")

    return cm, cm_normalized


cm, cm_normalized = generate_and_plot_confusion_matrix(model, test_loader, device, ax3)

class_names = [str(i) for i in range(num_classes)]
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names, ax=ax4)
ax4.set_title('Raw Confusion Matrix (Test Set)')
ax4.set_xlabel('Predicted Class')
ax4.set_ylabel('True Class')

plt.tight_layout()
plt.show()


# -------------------- 6. Final Evaluation --------------------
def final_evaluation(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    final_acc = 100 * np.sum(all_preds == all_labels) / len(all_labels)

    print("\nRecognition accuracy for each class:")
    for class_id in range(num_classes):
        mask = all_labels == class_id
        if np.sum(mask) > 0:
            class_acc = 100 * np.sum(all_preds[mask] == class_id) / np.sum(mask)
            print(f"  Class {class_id}: {class_acc:.2f}%")

    print(f"\nFinal overall test accuracy: {final_acc:.2f}%")
    return final_acc


final_acc = final_evaluation(model, test_loader, device)


# Single-sample prediction example (number of features changed to 4)
def predict_hardness(model, sample, scaler, device):
    model.eval()
    sample_scaled = scaler.transform(sample.reshape(-1, sample.shape[-1])).reshape(1, sample.shape[0], sample.shape[1])
    sample_tensor = torch.tensor(sample_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        output = model(sample_tensor)
        prob = torch.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
    return pred.item(), prob.cpu().numpy()[0]


sample_idx = 10
if len(X_test) > sample_idx:
    sample = X_test[sample_idx]
    true_label = y_test[sample_idx]
    pred_label, pred_prob = predict_hardness(model, sample, scaler, device)
    print(f"\nSingle-sample prediction example:")
    print(f"  True class: {true_label}")
    print(f"  Predicted class: {pred_label}")
    print(f"  Class probabilities: {dict(zip(range(num_classes), np.round(pred_prob, 4)))}")


# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'feature_cols': ['CH0_R', 'CH1_R', 'CH2_R', 'CH3_R'],
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'num_classes': num_classes
}, 'hardness_bilstm_model_4ch.pth')
print("\nModel has been saved as: hardness_bilstm_model_4ch.pth")