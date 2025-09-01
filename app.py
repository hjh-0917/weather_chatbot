from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import joblib

# 🔹 모델 정의 (train_model.py와 동일)
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

# 🔹 Flask 앱
app = Flask(__name__)

# 🔹 모델 불러오기
model = OutfitNet(input_size=5, hidden_size=128, output_sizes=[0,0,0,0])  # 나중에 encoder로 크기 맞춤
X_encoders = joblib.load("X_encoder.pkl")
y_encoders = joblib.load("y_encoder.pkl")

for i, enc in enumerate(y_encoders):
    model.output_layers[i] = nn.Linear(128, len(enc.classes_))
model.load_state_dict(torch.load("outfit_model.pth"))
model.eval()

@app.route("/")
def index():
    return render_template("chat_select.html")

@app.route("/start", methods=["POST"])
def start():
    persona = request.form.get("persona")
    return jsonify({"ok": True})

@app.route("/chat", methods=["GET","POST"])
def chat():
    if request.method == "POST":
        persona = request.form.get("persona")
        style = request.form.get("style")
        tone = request.form.get("tone")
        shape = request.form.get("shape")
        weather = request.form.get("weather")

        # 입력 인코딩
        x_input = []
        for i, val in enumerate([persona, style, tone, shape, weather]):
            x_input.append(X_encoders[i].transform([val])[0])
        x_tensor = torch.tensor([x_input], dtype=torch.float32)

        # 예측
        with torch.no_grad():
            outputs = model(x_tensor)
        result = {}
        for i, output in enumerate(outputs):
            pred_idx = torch.argmax(output, dim=1).item()
            pred_val = y_encoders[i].inverse_transform([pred_idx])[0]
            result[["상의","하의","신발","액세서리"][i]] = pred_val
        return jsonify(result)
    else:
        return render_template("chat.html")
    
if __name__ == "__main__":
    app.run(debug=True)

