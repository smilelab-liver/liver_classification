import torch
import torch.nn as nn

class AreaClassifier(nn.Module):
    def __init__(self, num_types, embedding_dim):
        super(AreaClassifier, self).__init__()
        self.embedding = nn.Embedding(num_types, embedding_dim)  # 類型嵌入層
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, 3)  # 分級指數 0-4
        )

    def forward(self, x):
        # x 是形狀 [batch_size, 2, 100]
        types, areas = x[:, 0, :].long(), x[:, 1, :]  # 類型和面積分離
        embedded = self.embedding(types)  # [batch_size, 100, embedding_dim]
        weighted = embedded * areas.unsqueeze(-1)  # 面積加權嵌入 [batch_size, 100, embedding_dim]
        pooled = weighted.sum(dim=1) / (areas.sum(dim=1, keepdim=True) + 1e-8)  # 加權平均 [batch_size, embedding_dim]
        return self.fc(pooled)  # 經過全連接層輸出
