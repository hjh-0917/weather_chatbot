import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path
import random

# 🔹 데이터 로드
def load_data():
    dataset = []
    personas = ["teen_female", "teen_male", "twenties_female", "twenties_male", "thirties_female", "thirties_male"]
    styles = ["cute","gentle","luxury","practical","trendy"]

    for persona in personas:
        for style in styles:
            file_path = Path(f"data/{persona}/{style}.json")
            if not file_path.exists():
                continue
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 모든 체형, 피부톤, 날씨, 사이즈 데이터를 flatten
                for tone in ["웜톤","쿨톤"]:
                    for shape in ["마른","보통","통통","역삼각형"]:
                        for weather in ["더움","선선","추움"]:
                            outfits = data[style][persona.split("_")[1]][persona.split("_")[0]][weather][tone][shape]
                            for outfit in outfits:
                                # 입력: persona + style + tone + shape + weather
                                X = [persona, style, tone, shape, weather]
                                # 출력: 상의, 하의, 신발, 액세서리
                                y = [outfit["상의"], outfit["하의"], outfit["신발"], outfit["액세서리"]]
                                dataset.append((X, y))
    return dataset

# 🔹 Label Encoding
def encode_dataset(dataset):
    X_raw = [x for x,y in dataset]
    y_raw = [y for x,y in dataset]

    X_encoders = [LabelEncoder() for _ in range(5)]
    X_encoded = []
    for i, enc in enumerate(X_encoders):
        col = [row[i] for row in X_raw]
        X_encoded.append(enc.fit_transform(col))
    X_encoded = list(zip(*X_encoded))  # [(persona, style, tone, shape, weather), ...]

    y_encoders = [LabelEncoder() for _ in range(4)]
    y_encoded = []
    for i, enc in enumerate(y_encoders):
        col = [row[i] for row in y_raw]
        y_encoded.append(enc.fit_transform(col))
    y_encoded = list(zip(*y_encoded))  # [(상의, 하의, 신발, 액세서리), ...]

    return X_encoded, y_encoded, X_encoders, y_encoders

# 🔹 모델 정의 (다중 출력)
class OutfitNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_sizes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.output_layers = nn.ModuleList([nn.Linear(hidden_size, out_size) for out_size in output_sizes])

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        outputs = [layer(x) for layer in self.output_layers]
        return outputs

# 🔹 학습
def train():
    dataset = load_data()
    X_encoded, y_encoded, X_encoders, y_encoders = encode_dataset(dataset)

    # Tensor 변환
    X_tensor = torch.tensor(X_encoded, dtype=torch.float32)
    y_tensor = [torch.tensor([yi[i] for yi in y_encoded], dtype=torch.long) for i in range(4)]

    model = OutfitNet(input_size=5, hidden_size=128, output_sizes=[len(enc.classes_) for enc in y_encoders])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = sum([criterion(outputs[i], y_tensor[i]) for i in range(4)])
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/100 Loss: {loss.item():.4f}")

    # 🔹 저장
    torch.save(model.state_dict(), "outfit_model.pth")
    joblib.dump(X_encoders, "X_encoder.pkl")
    joblib.dump(y_encoders, "y_encoder.pkl")
    print("모델과 인코더 저장 완료!")

if __name__ == "__main__":
    train()
