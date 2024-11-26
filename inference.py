import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import AreaDataset,process_json_files,process_json_file
from model import AreaClassifier
import os

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './best_model.pth'


model = AreaClassifier(num_types=3, embedding_dim=8)  # 假設有 3 種類型（portal, central, zone2）
model = model.to(device)
model.load_state_dict(torch.load(MODEL_PATH))

model.eval()
with torch.no_grad():

    test_root_path = './test_json'

    total_num = {}
    total_correct = {}
    for label in os.listdir(test_root_path):
        test_path = os.path.join(test_root_path,label)
        total_num[label] = 0
        total_correct[label] = 0
        for file in os.listdir(test_path):
            test_file_path = os.path.join(test_path,file)
            data = process_json_file(test_file_path)
            features = data[0]["features"]
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            features_tensor = features_tensor.to(device)
            outputs = model(features_tensor)
            predictions = torch.argmax(outputs, dim=1)
            predictions_int = predictions.item() if predictions.numel() == 1 else predictions.tolist()
            if int(label) == predictions_int:
                total_correct[label] += 1
            total_num[label]+=1
    total_acc = {}
    for key, value in total_num.items():
        total_acc[key] = total_correct[key]/value
    print(total_acc)