import os
import json
import requests
from flask import Flask, render_template, request, session

app = Flask(__name__)
app.secret_key = "your_secret_key"

# OpenWeather API key (Render 환경변수에서 불러오기)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_KEY_HERE")

# 캐릭터 목록 (한글 이름 + 이미지 매칭)
CHAR_LABEL_KO = {
    "trendy": "트렌디 스타일러",
    "practical": "실용파 스타일러",
    "luxury": "럭셔리 스타일러",
    "gentle": "신사 스타일러",
    "cute": "큐트 스타일러",
}

CHAR_IMAGES = {
    "trendy": "images/trendy.png",
    "practical": "images/practical.png",
    "luxury": "images/luxury.png",
    "gentle": "images/gentle.png",
    "cute": "images/cute.png",
}


def get_weather(location):
    """지역 이름으로 현재 날씨 불러오기"""
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr"
    res = requests.get(url)
    data = res.json()
    if res.status_code != 200:
        return None

    temp = data["main"]["temp"]
    weather_desc = data["weather"][0]["description"]

    # 날씨 카테고리 분류
    if temp >= 27:
        category = "더움"
    elif 15 <= temp < 27:
        category = "선선"
    else:
        category = "추움"

    return temp, weather_desc, category


def load_outfit(age_group, gender, character, weather, skin_tone, body_shape):
    """조건에 맞는 JSON 파일 불러와 코디 추천"""
    path = os.path.join("JSON", f"{age_group}_{gender}", f"{character}.json")

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        outfit = data[character][gender][age_group][weather][skin_tone][body_shape]
        return outfit
    except KeyError:
        return None


@app.route("/")
def select_character():
    return render_template("chat_select.html", characters=CHAR_LABEL_KO, images=CHAR_IMAGES)


@app.route("/start", methods=["POST"])
def start():
    session.clear()
    session["character"] = request.form["character"]
    # 첫 질문은 챗봇이 먼저
    return "안녕하세요! 코디 추천을 위해 지역(도시 이름)을 알려주세요 😊"


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]

    # 입력 단계별 저장
    if "location" not in session:
        session["location"] = user_message
        return "성별을 알려주세요 (남성/여성)"
    elif "gender" not in session:
        session["gender"] = "male" if "남" in user_message else "female"
        return "나이를 알려주세요 (예: 16)"
    elif "age" not in session:
        age = int(user_message)
        session["age"] = age
        if age < 20:
            session["age_group"] = "teen"
        else:
            session["age_group"] = "twenties"
        return "당신의 체형을 알려주세요 (예시: 마른, 보통, 통통, 역삼각형)"
    elif "body_shape" not in session:
        session["body_shape"] = user_message.strip()
        return "당신의 피부 톤을 알려주세요 (예시: 웜톤, 쿨톤)"
    elif "skin_tone" not in session:
        session["skin_tone"] = user_message.strip()

        # 최종 단계: 코디 추천
        location = session["location"]
        gender = session["gender"]
        age_group = session["age_group"]
        character = session["character"]
        body_shape = session["body_shape"]
        skin_tone = session["skin_tone"]

        weather_info = get_weather(location)
        if not weather_info:
            return "날씨 정보를 가져오지 못했습니다. 다시 시도해주세요."

        temp, desc, category = weather_info

        outfit = load_outfit(age_group, gender, character, category, skin_tone, body_shape)

        if not outfit:
            return f"{location} · {temp:.1f}°C · '{desc}' → 조건에 맞는 코디 데이터를 찾을 수 없습니다."

        result = (
            f"{location} · {temp:.1f}°C · '{desc}'\n"
            f"{session['age']}대 {'남성' if gender == 'male' else '여성'} · {category}\n"
            f"컨셉: {CHAR_LABEL_KO[character]}\n"
            f"피부톤: {skin_tone}, 체형: {body_shape}\n"
            f"→ 상의: {outfit['상의']}, 하의: {outfit['하의']}, 신발: {outfit['신발']}, 액세서리: {outfit['액세서리']}"
        )
        return result

    return "입력을 다시 확인해주세요."


# -----------------------
# Flask 실행 시작점
# -----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
