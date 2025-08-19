from flask import Flask, render_template, request, jsonify, session
import os, json, requests

app = Flask(__name__)
app.secret_key = "your_secret_key"  # 세션에 필요 (배포시 안전하게 변경)

# 한국어 → 영어 도시 변환
korean_to_english = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu", "인천": "Incheon",
    "광주": "Gwangju", "대전": "Daejeon", "울산": "Ulsan", "세종": "Sejong",
    "경기": "Gyeonggi-do", "강원": "Gangwon-do", "충북": "Chungcheongbuk-do",
    "충남": "Chungcheongnam-do", "전북": "Jeollabuk-do", "전남": "Jeollanam-do",
    "경북": "Gyeongsangbuk-do", "경남": "Gyeongsangnam-do", "제주": "Jeju"
}

API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# 캐릭터 기본 정보
characters = {
    "trendy": "트렌디",
    "practical": "실용적",
    "luxury": "고급스러움",
    "cute": "귀여움",
    "gentle": "신사"
}

@app.route("/")
def select():
    return render_template("chat_select.html", characters=characters)

@app.route("/chat/<character>", methods=["GET", "POST"])
def chat(character):
    return render_template("chat.html", character=character, characters=characters)

# ✅ 날씨 가져오기
def get_weather(city_kor):
    city_eng = korean_to_english.get(city_kor)
    if not city_eng:
        return None, None
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_eng}&appid={API_KEY}&units=metric&lang=kr"
    res = requests.get(url)
    if res.status_code != 200:
        return None, None
    data = res.json()
    return data["weather"][0]["description"], data["main"]["temp"]

# ✅ 챗봇 대화 처리
@app.route("/start", methods=["POST"])
def start():
    data = request.json
    session["gender"] = data["gender"]
    session["age"] = int(data["age"])
    session["city"] = data["city"]
    session["body_shape"] = data.get("body_shape", "보통")
    session["skin_tone"] = data.get("skin_tone", "웜톤")

    weather, temp = get_weather(session["city"])
    session["weather"] = weather
    session["temperature"] = temp

    return jsonify({"reply": f"{session['city']}의 날씨는 {weather}, {temp}°C 입니다. 체형과 피부톤도 반영해 코디를 추천해드릴게요."})

@app.route("/chat", methods=["POST"])
def chat_api():
    character = request.json["character"]
    gender = session.get("gender")
    age = session.get("age")
    city = session.get("city")
    weather = session.get("weather")
    temp = session.get("temperature")
    body_shape = session.get("body_shape")
    skin_tone = session.get("skin_tone")

    # 나이대 분류
    if age < 20:
        age_group = "teen"
    else:
        age_group = "twenties"

    gender_group = "male" if "남" in gender else "female"

    # 날씨 분류
    if temp >= 25:
        temp_group = "더움"
    elif temp >= 15:
        temp_group = "선선"
    else:
        temp_group = "추움"

    # JSON 파일 경로
    json_path = f"JSON/{age_group}_{gender_group}/{character}.json"

    if not os.path.exists(json_path):
        return jsonify({"reply": "해당 조건에 맞는 코디 데이터가 아직 준비되지 않았습니다."})

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        outfit = data[character][gender_group][f"{age_group == 'teen' and '10대' or '20대'}"][temp_group][skin_tone][body_shape]
        reply = f"""[추천 코디 - {characters[character]} 스타일]  
- 상의: {outfit['상의']}  
- 하의: {outfit['하의']}  
- 신발: {outfit['신발']}  
- 액세서리: {outfit['액세서리']}"""
    except Exception as e:
        reply = "조건에 맞는 코디를 찾을 수 없습니다. JSON 데이터를 확인해주세요."

    return jsonify({"reply": reply})

# ✅ 저장 기능
@app.route("/save", methods=["POST"])
def save():
    outfit = request.json.get("outfit")
    if "saved_outfits" not in session:
        session["saved_outfits"] = []
    session["saved_outfits"].append(outfit)
    session.modified = True
    return jsonify({"status": "ok"})

@app.route("/saved")
def saved():
    outfits = session.get("saved_outfits", [])
    return render_template("saved.html", outfits=outfits)

# ✅ 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

