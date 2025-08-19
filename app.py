from flask import Flask, render_template, request, session, jsonify
import os
import json
import requests

app = Flask(__name__)
app.secret_key = "your_secret_key"

# 캐릭터 설정
characters = {
    "trendy": "트렌디",
    "practical": "실용적",
    "luxury": "럭셔리",
    "gentle": "신사",
    "cute": "귀여움"
}

# ✅ 날씨 API
API_KEY = os.environ.get("OPENWEATHER_API_KEY")

def get_weather_condition(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric&lang=kr"
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    temp = data['main']['temp']
    if temp >= 23:
        return "더움"
    elif 10 <= temp < 23:
        return "선선"
    else:
        return "추움"

# ✅ 캐릭터 선택 페이지
@app.route("/")
def select():
    return render_template("chat_select.html", characters=characters)

# ✅ 캐릭터 선택 결과 → 챗봇 시작
@app.route("/start", methods=["POST"])
def start():
    character = request.form.get("character")  # ✅ form으로 받기
    session["character"] = character
    return render_template("chat.html", character=character, character_label=characters[character])

# ✅ 챗봇 대화 처리
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message = data.get("message", "")
    character = session.get("character", "trendy")

    # 사용자 입력 파싱
    if "지역:" in message and "성별:" in message and "나이:" in message and "체형:" in message and "피부톤:" in message:
        try:
            parts = dict(item.split(":") for item in message.split(","))
            city = parts.get("지역").strip()
            gender = parts.get("성별").strip()
            age = parts.get("나이").strip()
            body_type = parts.get("체형").strip()
            skin_tone = parts.get("피부톤").strip()
        except:
            return jsonify({"response": "입력 형식이 올바르지 않아요. 예시: 지역: 서울, 성별: 남자, 나이: 10대, 체형: 마른, 피부톤: 웜톤"})

        # 날씨 가져오기
        condition = get_weather_condition(city)
        if not condition:
            return jsonify({"response": f"{city}의 날씨 정보를 불러올 수 없어요."})

        # JSON 파일 경로 구성
        folder = None
        if "10" in age:
            folder = f"teen_{'male' if gender == '남자' else 'female'}"
        elif "20" in age:
            folder = f"twenties_{'male' if gender == '남자' else 'female'}"

        json_path = os.path.join("JSON", folder, f"{character}.json")
        if not os.path.exists(json_path):
            return jsonify({"response": "해당 조건에 맞는 코디 정보를 찾을 수 없어요."})

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        try:
            coordi = data[character][gender][age][condition][skin_tone][body_type]
            response_text = (
                f"👕 상의: {coordi['상의']}\n"
                f"👖 하의: {coordi['하의']}\n"
                f"👟 신발: {coordi['신발']}\n"
                f"🎒 액세서리: {coordi['액세서리']}"
            )
        except KeyError:
            response_text = "조건에 맞는 코디를 찾지 못했어요."

        return jsonify({"response": response_text})

    else:
        return jsonify({"response": "안녕하세요! 코디 추천을 위해 '지역, 성별, 나이, 체형, 피부톤'을 입력해주세요.\n예시: 지역: 서울, 성별: 남자, 나이: 10대, 체형: 마른, 피부톤: 웜톤"})

# ✅ Render 배포용
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

