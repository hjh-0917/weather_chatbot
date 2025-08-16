from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# JSON 파일 루트 경로
JSON_ROOT = "json"

# 선택 가능한 옵션들
AGES = ["10대", "20대", "30대"]
GENDERS = ["male", "female"]
STYLES = ["trendy", "practical", "luxury", "gentle", "cute"]
WEATHERS = ["더움", "선선", "추움"]
TONES = ["웜톤", "쿨톤"]
BODIES = ["마른", "보통", "통통", "역삼각형"]

def load_outfit(age, gender, style):
    """해당 age/gender/style 경로의 JSON 파일 불러오기"""
    filepath = os.path.join(JSON_ROOT, f"{age}_{gender}", f"{style}.json")
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_outfit(data, weather, tone, body):
    """JSON 데이터에서 날씨, 피부톤, 체형에 맞는 코디 반환"""
    try:
        # JSON 내부 구조 탐색
        style_key = list(data.keys())[0]
        age_gender_data = data[style_key]
        # 성별/연령대는 이미 JSON 파일 이름으로 구분되어 있으므로 data 안에 들어있지 않을 수 있음
        outfit = age_gender_data
        if weather in outfit:
            outfit = outfit[weather]
        if tone in outfit:
            outfit = outfit[tone]
        if body in outfit:
            return outfit[body]
        return None
    except Exception as e:
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = request.form.get("age")
        gender = request.form.get("gender")
        style = request.form.get("style")
        weather = request.form.get("weather")
        tone = request.form.get("tone")
        body = request.form.get("body")

        data = load_outfit(age, gender, style)
        if not data:
            return "해당 코디 데이터가 없습니다."

        outfit = get_outfit(data, weather, tone, body)
        if not outfit:
            return "선택한 조건에 맞는 코디가 없습니다."

        return render_template("chat.html", outfit=outfit)

    return render_template(
        "chat_select.html",
        ages=AGES,
        genders=GENDERS,
        styles=STYLES,
        weathers=WEATHERS,
        tones=TONES,
        bodies=BODIES
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
