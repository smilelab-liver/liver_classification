import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import AreaDataset,process_json_files,process_json_file
from model import AreaClassifier

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())

print('GPU state:', device)


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 檢查設備
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
print('GPU state:', device)

# 處理 JSON 數據
json_root_path = "./json"
batch_size = 4
epochs = 10000

processed_data = process_json_files(json_root_path, input_length=100)

# 切分訓練集和驗證集
train_data, val_data = train_test_split(processed_data, test_size=0.1, random_state=42)
train_dataset = AreaDataset(train_data, input_length=50)
val_dataset = AreaDataset(val_data, input_length=50)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 創建模型
model = AreaClassifier(num_types=3, embedding_dim=8)  # 假設有 3 種類型（portal, central, zone2）
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 添加學習率調度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# 保存最佳模型
best_val_loss = float('inf')
best_model_path = "best_model.pth"

for epoch in range(epochs):
    # 訓練階段
    model.train()
    train_loss = 0
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # 驗證階段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    # 動態更新學習率
    scheduler.step(val_loss)

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

    # 打印當前 epoch 的訓練和驗證損失
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
    print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")
