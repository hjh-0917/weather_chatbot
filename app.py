# app.py (최종본: JSON 조회 + 도시 변환 + OpenWeather + ML 추론 통합)
import os
import json
import random
from pathlib import Path
from typing import Tuple, Optional

import requests
import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, session, jsonify

# -----------------------
# 설정 및 상수
# -----------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "").strip()

# 한글 도시명 → 영문(일부 대표 매핑)
KOR_TO_ENG_CITY = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu", "인천": "Incheon",
    "광주": "Gwangju", "대전": "Daejeon", "울산": "Ulsan", "세종": "Sejong",
    "경기": "Gyeonggi-do", "강원": "Gangwon-do", "충북": "Chungcheongbuk-do",
    "충남": "Chungcheongnam-do", "전북": "Jeollabuk-do", "전남": "Jeollanam-do",
    "경북": "Gyeongsangbuk-do", "경남": "Gyeongsangnam-do", "제주": "Jeju"
}

# UI용 캐릭터 라벨/이미지
PERSONA_LABEL = {
    "trendy": "트렌디",
    "practical": "실용파",
    "luxury": "럭셔리",
    "gentle": "신사",
    "cute": "귀여움"
}
PERSONA_IMAGE = {
    "trendy": "trendy.png",
    "practical": "practical.png",
    "luxury": "luxury.png",
    "gentle": "gentle.png",
    "cute": "cute.png"
}

# JSON 폴더 위치 (프로젝트 루트 / JSON)
JSON_ROOT = Path(__file__).parent / "JSON"
MODELS_DIR = Path(__file__).parent / "models"

# 대화 단계
STEPS = ["location", "gender", "age", "skin_tone", "body_shape"]

# -----------------------
# 유틸 함수들
# -----------------------
def normalize_gender(text: str) -> str:
    t = text.strip().lower()
    if t in ["남", "남자", "male", "m", "boy"]:
        return "male"
    if t in ["여", "여자", "female", "f", "girl"]:
        return "female"
    return ""

def age_to_label_and_dir(age: int, gender: str) -> Tuple[str, str]:
    if age <= 19:
        return f"teen_{gender}", "10대"
    if 20 <= age <= 29:
        return f"twenties_{gender}", "20대"
    if 30 <= age <= 39:
        return f"thirties_{gender}", "30대"
    return f"thirties_{gender}", "30대"

def temp_to_category(temp_c: float) -> str:
    if temp_c >= 24:
        return "더움"
    if temp_c >= 15:
        return "선선"
    return "추움"

def safe_pick(d: dict, key: str):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        if d:
            first_key = next(iter(d.keys()))
            return d[first_key]
    return None

def pick_from_json(json_path: Path, persona: str, gender_norm: str, age_label: str,
                   weather_cat: str, skin: str, body: str):
    default_skin = "웜톤"
    default_body = "보통"

    if not json_path.exists():
        return None

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    node = safe_pick(data, persona)
    node = safe_pick(node, gender_norm)
    node = safe_pick(node, age_label)
    node = safe_pick(node, weather_cat)
    node_skin = safe_pick(node, skin)
    if node_skin is None:
        node_skin = safe_pick(node, default_skin)
    outfit = safe_pick(node_skin, body)
    if outfit is None:
        outfit = safe_pick(node_skin, default_body)

    # outfit can be dict or list(여러 코디)
    if isinstance(outfit, list):
        # 무작위 또는 첫번째 선택 — 현재는 무작위로 하나 선택
        chosen = random.choice(outfit) if outfit else None
        return chosen if isinstance(chosen, dict) else None
    if isinstance(outfit, dict):
        return outfit
    return None

def get_weather(city_kor: str) -> Tuple[Optional[float], Optional[str]]:
    """
    한글 도시명 → 영문으로 변환 후 OpenWeather에서 현재 기온과 설명을 반환.
    실패시 (None, None)
    """
    city_query = KOR_TO_ENG_CITY.get(city_kor.strip(), city_kor.strip())
    if not OPENWEATHER_API_KEY:
        return None, None
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city_query}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr"
        )
        r = requests.get(url, timeout=8)
        data = r.json()
        if r.status_code == 200 and "main" in data:
            temp = float(data["main"]["temp"])
            desc = data["weather"][0]["description"]
            return temp, desc
    except Exception:
        pass
    return None, None

# -----------------------
# ML 모델 (멀티헤드) 정의 및 로드
# -----------------------
class OutfitNet(nn.Module):
    """
    멀티헤드 모델: 각 헤드(상의/하의/신발/액세서리)를 독립 분류로 출력
    head_sizes: dict: {"상의": n_top_classes, ...}
    """
    def __init__(self, input_dim, hidden_dim, head_sizes):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.heads = nn.ModuleDict()
        for k, sz in head_sizes.items():
            self.heads[k] = nn.Linear(hidden_dim, sz)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        out = {}
        for k, layer in self.heads.items():
            out[k] = layer(x)
        return out

# 전역 ML 리소스 컨테이너
ML_RES = {"encoders": None, "meta": None, "model": None, "device": None}

def load_model_resources(hidden_dim: int = 256):
    enc_path = MODELS_DIR / "encoders.joblib"
    meta_path = MODELS_DIR / "meta.joblib"
    model_path = MODELS_DIR / "outfit_net.pt"
    if not enc_path.exists() or not meta_path.exists() or not model_path.exists():
        print("ML 리소스가 models/에 없습니다. ML 추론은 비활성화됩니다.")
        return
    try:
        encoders = joblib.load(enc_path)
        meta = joblib.load(meta_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # input_dim: meta 의 onehot_feature_names 길이
        feature_names = meta.get("onehot_feature_names") or []
        input_dim = len(feature_names)
        head_sizes = {k: len(v) for k, v in meta.get("head_classes", {}).items()}
        model = OutfitNet(input_dim=input_dim, hidden_dim=hidden_dim, head_sizes=head_sizes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        ML_RES.update({"encoders": encoders, "meta": meta, "model": model, "device": device})
        print("✅ ML 리소스 로드 완료. 추론 사용 가능.")
    except Exception as e:
        print("ML 리소스 로드 실패:", e)

# 앱 시작 시 ML 리소스 시도 로드
load_model_resources()

def predict_outfit_with_model(features: dict, topk: int = 1):
    """
    features: dict with keys persona, gender, age_label, weather_cat, skin, body
    returns: dict mapping '상의','하의','신발','액세서리' -> predicted label (문자열) or None
    """
    if ML_RES["model"] is None or ML_RES["encoders"] is None:
        return None

    enc = ML_RES["encoders"]
    meta = ML_RES["meta"]
    model = ML_RES["model"]
    device = ML_RES["device"]

    # Create DataFrame for one sample (order must match training)
    df = pd.DataFrame([{
        "persona": features.get("persona", ""),
        "gender": features.get("gender", ""),
        "age_label": features.get("age_label", ""),
        "weather_cat": features.get("weather_cat", ""),
        "skin": features.get("skin", ""),
        "body": features.get("body", "")
    }])
    try:
        X = enc['onehot'].transform(df[["persona","gender","age_label","weather_cat","skin","body"]]).toarray().astype('float32')
    except Exception as e:
        # 인코더 차이 등으로 실패할 수 있음
        print("OneHot transform 실패:", e)
        return None

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(X_tensor)  # dict of tensors, shape (1, classes)
    preds = {}
    for head, logit in logits.items():
        idx = int(torch.argmax(logit, dim=1).cpu().numpy()[0])
        classes = meta["head_classes"].get(head, [])
        label = classes[idx] if idx < len(classes) else ""
        preds[head] = label

    # Map to Korean key names used by app
    return {
        "상의": preds.get("상의", ""),
        "하의": preds.get("하의", ""),
        "신발": preds.get("신발", ""),
        "액세서리": preds.get("액세서리", "")
    }

# -----------------------
# 라우트: 선택 화면 / 채팅 화면 / 채팅 API
# -----------------------
@app.route("/", methods=["GET"])
def select():
    return render_template("chat_select.html", persona_label=PERSONA_LABEL, persona_image=PERSONA_IMAGE)

@app.route("/start", methods=["POST"])
def start():
    persona = request.form.get("persona", "").strip()
    if persona not in PERSONA_LABEL:
        return jsonify({"ok": False, "error": "캐릭터 선택이 올바르지 않습니다."}), 400
    session.clear()
    session["persona"] = persona
    session["step"] = 0
    session["answers"] = {}
    return jsonify({"ok": True})

@app.route("/chat", methods=["GET"])
def chat_page():
    persona = session.get("persona", "")
    label = PERSONA_LABEL.get(persona, persona)
    image = PERSONA_IMAGE.get(persona, "")
    return render_template("chat.html", persona=persona, persona_label=label, persona_image=image)

@app.route("/chat", methods=["POST"])
def chat_api():
    step_idx = session.get("step", 0)
    answers = session.get("answers", {})
    persona = session.get("persona", "")

    user_msg = (request.json or {}).get("message", "").strip()

    # 단계별 흐름은 원래 구현과 동일
    if step_idx == 0:
        if not user_msg:
            return jsonify({"reply": "어느 지역에 계신가요? (예: 서울, 부산)"})
        answers["location"] = user_msg
        session["step"] = 1
        session["answers"] = answers
        return jsonify({"reply": "성별을 알려주세요. (남/여)"})

    if step_idx == 1:
        g = normalize_gender(user_msg)
        if not g:
            return jsonify({"reply": "성별을 남/여 로 입력해주세요."})
        answers["gender"] = g
        session["step"] = 2
        session["answers"] = answers
        return jsonify({"reply": "나이를 숫자로 입력해주세요. (예: 17, 22, 35)"})

    if step_idx == 2:
        try:
            age = int(user_msg)
            if age <= 0 or age > 120:
                raise ValueError
        except Exception:
            return jsonify({"reply": "나이는 1~120 사이의 숫자로 입력해주세요."})
        answers["age"] = age
        session["step"] = 3
        session["answers"] = answers
        return jsonify({"reply": "피부 톤을 알려주세요. (웜톤/쿨톤)"})

    if step_idx == 3:
        tone = user_msg.replace("톤", "").strip()
        if tone not in ["웜", "쿨", "웜톤", "쿨톤"]:
            return jsonify({"reply": "피부 톤은 웜톤/쿨톤 중에서 선택해주세요."})
        answers["skin_tone"] = "웜톤" if "웜" in tone else "쿨톤"
        session["step"] = 4
        session["answers"] = answers
        return jsonify({"reply": "체형을 알려주세요. (마른/보통/통통/역삼각형)"})

    if step_idx == 4:
        if user_msg not in ["마른", "보통", "통통", "역삼각형"]:
            return jsonify({"reply": "체형은 마른/보통/통통/역삼각형 중에서 입력해주세요."})
        answers["body_shape"] = user_msg

        # 날씨 조회 (도시명 한글→영문 변환 포함)
        temp, desc = get_weather(answers["location"])
        if temp is not None:
            weather_cat = temp_to_category(temp)
        else:
            weather_cat = "선선"
            desc = "날씨 조회 실패(임시로 선선 적용)"

        age_dir, age_label = age_to_label_and_dir(int(answers["age"]), answers["gender"])

        json_file = JSON_ROOT / age_dir / f"{persona}.json"
        outfit = pick_from_json(
            json_file,
            persona=persona,
            gender_norm=answers["gender"],
            age_label=age_label,
            weather_cat=weather_cat,
            skin=answers["skin_tone"],
            body=answers["body_shape"]
        )

        loc = answers["location"]
        weather_line = f"{loc}의 현재 날씨: {desc if desc else '정보 없음'}"

        # JSON 조회 실패 시 ML 예측 시도
        if not outfit:
            features = {
                "persona": persona,
                "gender": answers["gender"],
                "age_label": age_label,
                "weather_cat": weather_cat,
                "skin": answers["skin_tone"],
                "body": answers["body_shape"]
            }
            ml_outfit = predict_outfit_with_model(features)
            if ml_outfit:
                outfit = ml_outfit
                reply = (
                    f"(JSON에서 직접 찾을 수 없어 학습된 모델로 예측한 결과입니다.)\n"
                    f"{weather_line}\n"
                    f"(분류: {weather_cat}, 나이대: {age_label}, 피부톤: {answers['skin_tone']}, 체형: {answers['body_shape']})\n\n"
                    f"[예측된 코디]\n"
                    f"- 상의: {outfit.get('상의','')}\n"
                    f"- 하의: {outfit.get('하의','')}\n"
                    f"- 신발: {outfit.get('신발','')}\n"
                    f"- 액세서리: {outfit.get('액세서리','')}"
                )
            else:
                reply = (
                    f"선택 조합에 대한 코디 데이터를 찾지 못했습니다. "
                    f"(캐릭터: {PERSONA_LABEL.get(persona, persona)}, 성별: {answers['gender']}, "
                    f"나이: {answers['age']}, 날씨: {weather_cat}, 피부톤: {answers['skin_tone']}, 체형: {answers['body_shape']})\n"
                    f"JSON 파일 또는 모델을 확인해주세요."
                )
        else:
            outfit_line = (
                f"[오늘의 코디]\n"
                f"- 상의: {outfit.get('상의','')}\n"
                f"- 하의: {outfit.get('하의','')}\n"
                f"- 신발: {outfit.get('신발','')}\n"
                f"- 액세서리: {outfit.get('액세서리','')}"
            )
            reply = (
                f"{weather_line}\n"
                f"(분류: {weather_cat}, 나이대: {age_label}, 피부톤: {answers['skin_tone']}, 체형: {answers['body_shape']})\n\n"
                f"{outfit_line}"
            )

        session["step"] = 0
        session["answers"] = {}
        return jsonify({"reply": reply})

    # 기본 재시작
    session["step"] = 0
    session["answers"] = {}
    return jsonify({"reply": "대화를 다시 시작합니다. 어느 지역에 계신가요? (예: 서울, 부산)"})

# -----------------------
# 실행
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)



