from flask import Flask, render_template, request, session, jsonify
import requests
import os
import random

app = Flask(__name__)
app.secret_key = "secret-key"

API_KEY = os.environ.get("OPENWEATHER_API_KEY", "여기에_API키")

# 한글 → 영어 도시명
korean_to_english = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu",
    "인천": "Incheon", "광주": "Gwangju", "대전": "Daejeon",
    "울산": "Ulsan", "세종": "Sejong", "제주": "Jeju"
}

# 챗봇 캐릭터
characters = {
    "trendy": "트렌디 패션 전문가",
    "practical": "실속파 코디 장인",
    "luxury": "럭셔리 스타일리스트",
    "cute": "귀여운 패션 친구",
    "gentle": "신사 스타일 챗봇"
}

# 연령대 그룹화
def get_age_group(age):
    if age < 20: return "10대"
    elif age < 30: return "20대"
    elif age < 40: return "30대"
    else: return "40대 이상"

# 코디 추천
def get_outfit(temp, gender, age_group, character_key):
    outfit_data = {
        "gentle": {
            "hot": [
                f"{age_group} {gender}께는 시원한 린넨 셔츠와 슬림핏 슬랙스를 권해드립니다.",
                f"{age_group} {gender}께는 화이트 셔츠와 연한 그레이 슬랙스가 잘 어울립니다.",
                f"{age_group} {gender}께는 블루 스트라이프 셔츠와 면바지가 멋스럽습니다."
            ],
            "warm": [
                f"{age_group} {gender}께는 네이비 블레이저와 면바지를 추천드립니다.",
                f"{age_group} {gender}께는 얇은 카디건과 셔츠 조합이 좋습니다.",
                f"{age_group} {gender}께는 체크 셔츠와 다크 진이 고급스럽습니다."
            ],
            "cool": [
                f"{age_group} {gender}께는 트렌치코트와 니트 조합이 잘 어울립니다.",
                f"{age_group} {gender}께는 울 니트와 슬랙스를 추천드립니다.",
                f"{age_group} {gender}께는 가디건과 셔츠 조합이 좋습니다."
            ],
            "cold": [
                f"{age_group} {gender}께는 울 코트와 머플러를 추천드립니다.",
                f"{age_group} {gender}께는 더블 브레스트 코트와 니트 조합이 멋집니다.",
                f"{age_group} {gender}께는 두꺼운 체스터필드 코트가 잘 어울립니다."
            ]
        },
        "trendy": {
            "hot": [
                f"{age_group} {gender}에게는 오버핏 반팔 티셔츠와 조거 팬츠를 추천합니다.",
                f"{age_group} {gender}에게는 나시와 카고 팬츠 조합이 멋집니다.",
                f"{age_group} {gender}에게는 루즈핏 반팔과 데님 반바지가 좋습니다."
            ],
            "warm": [
                f"{age_group} {gender}에게는 루즈핏 셔츠와 데님 팬츠가 잘 어울립니다.",
                f"{age_group} {gender}에게는 후드 집업과 조거 팬츠 조합을 추천합니다.",
                f"{age_group} {gender}에게는 카디건과 트렌디한 청바지가 좋습니다."
            ],
            "cool": [
                f"{age_group} {gender}에게는 오버핏 후드와 와이드 팬츠가 좋습니다.",
                f"{age_group} {gender}에게는 항공 점퍼와 슬림 진 조합이 멋집니다.",
                f"{age_group} {gender}에게는 데님 자켓과 조거 팬츠가 잘 어울립니다."
            ],
            "cold": [
                f"{age_group} {gender}에게는 롱패딩과 조거 팬츠를 추천합니다.",
                f"{age_group} {gender}에게는 두꺼운 후드와 패딩 베스트 조합이 좋습니다.",
                f"{age_group} {gender}에게는 숏패딩과 카고 팬츠가 멋집니다."
            ]
        },
        "practical": {
            "hot": [
                f"{age_group} {gender}에게는 반팔 티셔츠와 반바지를 추천합니다.",
                f"{age_group} {gender}에게는 얇은 린넨 셔츠와 면바지가 좋습니다.",
                f"{age_group} {gender}에게는 기능성 반팔과 경량 반바지를 권해드립니다."
            ],
            "warm": [
                f"{age_group} {gender}에게는 얇은 긴팔 티셔츠와 면바지를 추천합니다.",
                f"{age_group} {gender}에게는 셔츠와 치노 팬츠 조합이 좋습니다.",
                f"{age_group} {gender}에게는 라이트 자켓과 데님 팬츠가 적당합니다."
            ],
            "cool": [
                f"{age_group} {gender}에게는 니트와 청바지를 추천합니다.",
                f"{age_group} {gender}에게는 후드와 면바지가 편안합니다.",
                f"{age_group} {gender}에게는 가디건과 셔츠 조합이 좋습니다."
            ],
            "cold": [
                f"{age_group} {gender}에게는 두꺼운 코트와 장갑을 추천합니다.",
                f"{age_group} {gender}에게는 경량 패딩과 머플러 조합이 좋습니다.",
                f"{age_group} {gender}에게는 방한 점퍼와 기모 바지를 권해드립니다."
            ]
        },
        "luxury": {
            "hot": [
                f"{age_group} {gender}에게는 구찌 반팔 셔츠와 로에베 샌들을 추천드립니다.",
                f"{age_group} {gender}에게는 프라다 셔츠와 보테가 베네타 로퍼가 잘 어울립니다.",
                f"{age_group} {gender}에게는 루이비통 폴로 셔츠와 발렌시아가 스니커즈가 멋집니다."
            ],
            "warm": [
                f"{age_group} {gender}에게는 버버리 셔츠와 톰포드 슬랙스를 추천드립니다.",
                f"{age_group} {gender}에게는 로로피아나 니트와 에르메스 로퍼가 잘 어울립니다.",
                f"{age_group} {gender}에게는 톰브라운 가디건과 보테가 베네타 팬츠가 좋습니다."
            ],
            "cool": [
                f"{age_group} {gender}에게는 발렌시아가 니트와 보테가 베네타 팬츠를 추천드립니다.",
                f"{age_group} {gender}에게는 디올 울 코트와 구찌 로퍼가 잘 어울립니다.",
                f"{age_group} {gender}에게는 루이비통 점퍼와 톰포드 팬츠가 좋습니다."
            ],
            "cold": [
                f"{age_group} {gender}에게는 몽클레어 코트와 루이비통 머플러를 추천드립니다.",
                f"{age_group} {gender}에게는 프라다 패딩과 보테가 베네타 부츠가 좋습니다.",
                f"{age_group} {gender}에게는 톰포드 울 코트와 에르메스 장갑이 잘 어울립니다."
            ]
        },
        "cute": {
            "hot": [
                f"{age_group} {gender}에게는 반팔 원피스와 샌들을 추천합니다.",
                f"{age_group} {gender}에게는 민소매 티셔츠와 플레어 스커트가 좋습니다.",
                f"{age_group} {gender}에게는 얇은 셔츠와 반바지가 잘 어울립니다."
            ],
            "warm": [
                f"{age_group} {gender}에게는 가디건과 플레어 스커트를 추천합니다.",
                f"{age_group} {gender}에게는 셔츠와 A라인 스커트 조합이 좋습니다.",
                f"{age_group} {gender}에게는 얇은 니트와 청바지가 잘 어울립니다."
            ],
            "cool": [
                f"{age_group} {gender}에게는 귀여운 니트와 치마를 추천합니다.",
                f"{age_group} {gender}에게는 후드티와 면바지가 잘 어울립니다.",
                f"{age_group} {gender}에게는 가디건과 청바지가 좋습니다."
            ],
            "cold": [
                f"{age_group} {gender}에게는 패딩과 머플러를 추천합니다.",
                f"{age_group} {gender}에게는 두꺼운 코트와 니트 원피스가 잘 어울립니다.",
                f"{age_group} {gender}에게는 후드 집업과 기모 바지가 좋습니다."
            ]
        }
    }

    # 온도 범위 → 키 매칭
    if temp >= 27:
        season_key = "hot"
    elif temp >= 20:
        season_key = "warm"
    elif temp >= 10:
        season_key = "cool"
    else:
        season_key = "cold"

    return random.choice(outfit_data[character_key][season_key])

@app.route("/")
def select_character():
    return render_template("chat_select.html", characters=characters)

@app.route("/start_chat", methods=["POST"])
def start_chat():
    session.clear()
    session["character"] = request.form.get("character")
    session["step"] = 1
    return render_template("chat.html", character=characters[session['character']])

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")
    step = session.get("step", 1)

    if step == 1:
        session["step"] = 2
        session["city"] = user_msg
        return jsonify({"reply": "성별을 입력해주세요 (남성/여성)"})
    elif step == 2:
        session["step"] = 3
        session["gender"] = user_msg
        return jsonify({"reply": "나이를 입력해주세요"})
    elif step == 3:
        try:
            age = int(user_msg)
        except:
            return jsonify({"reply": "나이는 숫자로 입력해주세요."})

        session["age"] = age
        session["step"] = 4

        city_kor = session["city"]
        city_eng = korean_to_english.get(city_kor)
        if not city_eng:
            return jsonify({"reply": f"'{city_kor}'는 지원하지 않는 지역입니다."})

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_eng}&appid={API_KEY}&units=metric&lang=kr"
        res = requests.get(url)
        if res.status_code != 200:
            return jsonify({"reply": "날씨 정보를 가져오지 못했습니다."})

        data = res.json()
        temp = data['main']['temp']
        weather_desc = data['weather'][0]['description']
        age_group = get_age_group(age)

        outfit = get_outfit(temp, session["gender"], age_group, session["character"])

        reply = f"{city_kor}의 현재 온도는 {temp}°C, 날씨는 '{weather_desc}'입니다.\n추천 코디: {outfit}"
        return jsonify({"reply": reply})
    else:
        return jsonify({"reply": "대화를 다시 시작하려면 페이지를 새로고침하세요."})

if __name__ == "__main__":
    app.run(debug=True)

