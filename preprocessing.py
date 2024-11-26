import json


import os
import json

def process_json_files(json_root_path, input_length=100):
    data = []
    type_mapping = {"portal": 0, "central": 1, "zone2": 2}  # 類型標籤映射

    # 遍歷資料夾（0, 1, 2）
    for label in os.listdir(json_root_path):
        label_path = os.path.join(json_root_path, label)
        if not os.path.isdir(label_path):
            continue

        # 遍歷每個 JSON 文件
        for json_file in os.listdir(label_path):
            if not json_file.endswith(".json"):
                continue

            json_path = os.path.join(label_path, json_file)

            with open(json_path, "r") as f:
                json_data = json.load(f)

            types = []
            areas = []

            # 處理 JSON 文件內容
            for key, value in json_data.items():
                type_name = key.split("#")[0]  # 提取類型名稱（例如 "portal"）
                type_id = type_mapping.get(type_name, -1)  # 映射類型標籤
                if type_id == -1:
                    continue  # 忽略未知類型

                types.append(type_id)
                areas.append(value)

            # 填充或截斷到固定長度
            # if len(types) < input_length:
            #     types.extend([0] * (input_length - len(types)))  # 類型填充 0
            #     areas.extend([0] * (input_length - len(areas)))  # 面積填充 0
            # else:
            #     types = types[:input_length]
            #     areas = areas[:input_length]

            # 添加到數據列表
            data.append({
                "features": [types, areas],
                "label": int(label)
            })

    return data


json_root_path = "./json"
processed_data = process_json_files(json_root_path, input_length=100)

# 查看處理後的數據
for sample in processed_data[:5]:  # 查看前 5 筆數據
    print(sample)
