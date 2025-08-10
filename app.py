from flask import Flask, render_template, request, jsonify
import requests
import os
import random

app = Flask(__name__)

# ✅ OpenWeather API 키
API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# ✅ 한글 → 영어 변환
korean_to_english = {
    "서울": "Seoul",
    "부산": "Busan",
    "대구": "Daegu",
    "인천": "Incheon",
    "광주": "Gwangju",
    "대전": "Daejeon",
    "울산": "Ulsan",
    "제주": "Jeju"
}

# ✅ 연령대별·날씨별 코디 추천 리스트
def build_outfit_list(temp, age_group):
    outfits = []

    if age_group == "청소년":  # 10대
        if temp >= 25:
            outfits = [
                ["루즈핏 반팔 티셔츠", "데님 반바지", "캔버스화"],
                ["그래픽 티셔츠", "조거 반바지", "운동화"],
                ["민소매 탑", "치마바지", "샌들"],
                ["크롭 반팔", "청반바지", "슬립온"],
                ["박시 셔츠", "린넨 반바지", "스니커즈"]
            ]
        elif temp >= 15:
            outfits = [
                ["맨투맨", "와이드 청바지", "운동화"],
                ["후드티", "슬림진", "캔버스화"],
                ["셔츠", "조거팬츠", "스니커즈"],
                ["카디건", "미디 스커트", "플랫슈즈"],
                ["롱슬리브 티셔츠", "면바지", "운동화"]
            ]
        elif temp >= 5:
            outfits = [
                ["니트", "슬랙스", "더비슈즈"],
                ["집업 후드", "청바지", "스니커즈"],
                ["기모 맨투맨", "조거팬츠", "운동화"],
                ["가디건", "롱스커트", "부츠"],
                ["후리스", "트레이닝 팬츠", "운동화"]
            ]
        else:
            outfits = [
                ["패딩", "기모 청바지", "부츠"],
                ["롱패딩", "조거팬츠", "운동화"],
                ["숏패딩", "와이드 팬츠", "부츠"],
                ["코트", "슬랙스", "로퍼"],
                ["점퍼", "청바지", "부츠"]
            ]

    elif age_group == "청년":  # 20~30대
        if temp >= 25:
            outfits = [
                ["린넨 셔츠", "슬림 치노 반바지", "로퍼"],
                ["반팔 셔츠", "슬랙스 반바지", "샌들"],
                ["헨리넥 티셔츠", "데님 반바지", "슬립온"],
                ["화이트 셔츠", "린넨 팬츠", "로퍼"],
                ["크루넥 티셔츠", "카고 반바지", "운동화"]
            ]
        elif temp >= 15:
            outfits = [
                ["니트", "슬랙스", "더비슈즈"],
                ["셔츠", "청바지", "로퍼"],
                ["맨투맨", "치노팬츠", "운동화"],
                ["트렌치코트", "슬랙스", "로퍼"],
                ["가디건", "데님 팬츠", "스니커즈"]
            ]
        elif temp >= 5:
            outfits = [
                ["코트", "슬랙스", "첼시부츠"],
                ["니트", "청바지", "로퍼"],
                ["가죽자켓", "블랙진", "부츠"],
                ["후드집업", "와이드팬츠", "운동화"],
                ["스웨터", "슬림진", "로퍼"]
            ]
        else:
            outfits = [
                ["롱패딩", "슬랙스", "부츠"],
                ["더블코트", "청바지", "첼시부츠"],
                ["숏패딩", "와이드팬츠", "운동화"],
                ["패딩베스트", "니트", "부츠"],
                ["무스탕", "슬림진", "부츠"]
            ]

    elif age_group in ["중년", "시니어"]:  # 40대 이상
        if temp >= 25:
            outfits = [
                ["린넨 셔츠", "일자 면바지", "로퍼"],
                ["반팔 카라티", "린넨 팬츠", "샌들"],
                ["화이트 셔츠", "베이지 슬랙스", "로퍼"],
                ["반팔 셔츠", "치노팬츠", "운동화"],
                ["린넨 자켓", "린넨 팬츠", "로퍼"]
            ]
        elif temp >= 15:
            outfits = [
                ["가디건", "슬랙스", "로퍼"],
                ["셔츠", "일자 청바지", "로퍼"],
                ["니트", "면바지", "로퍼"],
                ["트렌치코트", "슬랙스", "더비슈즈"],
                ["브이넥 니트", "청바지", "로퍼"]
            ]
        elif temp >= 5:
            outfits = [
                ["코트", "슬랙스", "부츠"],
                ["니트", "면바지", "로퍼"],
                ["패딩조끼", "셔츠", "슬랙스"],
                ["스웨터", "청바지", "운동화"],
                ["가죽자켓", "블랙진", "부츠"]
            ]
        else:
            outfits = [
                ["롱패딩", "슬랙스", "부츠"],
                ["숏패딩", "청바지", "운동화"],
                ["무스탕", "면바지", "부츠"],
                ["더블코트", "슬랙스", "로퍼"],
                ["코트", "청바지", "첼시부츠"]
            ]

    return random.choice(outfits)

# ✅ 날씨 가져오기
def get_weather(city_eng):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_eng}&appid={API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        data = res.json()
        return data["main"]["temp"], data["weather"][0]["main"].lower()
    return None, None

@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    session = request.json.get("session", {})

    if "step" not in session:
        session["step"] = 1
        return jsonify({"reply": "안녕하세요! 먼저 지역을 입력해주세요.", "session": session})

    if session["step"] == 1:
        session["city"] = user_input
        if session["city"] not in korean_to_english:
            return jsonify({"reply": "지원하지 않는 지역입니다. 다시 입력해주세요.", "session": session})
        session["step"] = 2
        return jsonify({"reply": "성별을 입력해주세요 (남성/여성).", "session": session})

    if session["step"] == 2:
        session["gender"] = user_input
        session["step"] = 3
        return jsonify({"reply": "나이를 입력해주세요.", "session": session})

    if session["step"] == 3:
        try:
            age = int(user_input)
            session["age"] = age
            if age < 20:
                session["age_group"] = "청소년"
            elif age < 40:
                session["age_group"] = "청년"
            else:
                session["age_group"] = "중년"
        except:
            return jsonify({"reply": "나이는 숫자로 입력해주세요.", "session": session})

        temp, weather_type = get_weather(korean_to_english[session["city"]])
        if temp is None:
            return jsonify({"reply": "날씨 정보를 가져오지 못했습니다.", "session": session})

        outfit = build_outfit_list(temp, session["age_group"])
        reply = f"{session['city']}의 현재 온도는 {temp}°C, 날씨는 {weather_type}입니다.\n추천 코디: {', '.join(outfit)}"

        # ✅ 배경 날씨 타입 추가
        session["weather_type"] = weather_type

        session["step"] = 1
        return jsonify({"reply": reply, "session": session})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
