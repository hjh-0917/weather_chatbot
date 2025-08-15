from flask import Flask, render_template, request, jsonify, session
import random
import requests
import os

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

# 캐릭터 목록
characters = {
    "trendy": "트렌디 전문가",
    "practical": "실속파 코디 장인",
    "luxury": "럭셔리 스타일리스트",
    "cute": "귀여운 패션 친구",
    "gentle": "신사 스타일 챗봇"
}

# 한글 → 영어 도시명 변환
korean_to_english = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu",
    "인천": "Incheon", "광주": "Gwangju", "대전": "Daejeon",
    "울산": "Ulsan", "세종": "Sejong", "경기": "Gyeonggi-do",
    "강원": "Gangwon-do", "충북": "Chungcheongbuk-do",
    "충남": "Chungcheongnam-do", "전북": "Jeollabuk-do",
    "전남": "Jeollanam-do", "경북": "Gyeongsangbuk-do",
    "경남": "Gyeongsangnam-do", "제주": "Jeju"
}

API_KEY = os.environ.get("OPENWEATHER_API_KEY")  # Render 환경변수

def get_weather(city_kor):
    city_eng = korean_to_english.get(city_kor)
    if not city_eng:
        return None, None
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_eng}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp'], data['weather'][0]['description']
    return None, None

def get_age_group(age):
    if age < 20: return "10대"
    elif age < 30: return "20대"
    elif age < 40: return "30대"
    else: return "40대 이상"

def get_outfit(character, gender, age, temp):
    age_group = get_age_group(age)

    outfits = {
        "trendy": {
            "hot": [
                f"{age_group} {gender}님께는 화이트 셔츠와 린넨 반바지, 샌들을 추천드립니다.",
                f"{age_group} {gender}님은 오버사이즈 티셔츠와 와이드 팬츠로 시원하게 입어보세요.",
                f"{age_group} {gender}님께는 민소매 셔츠와 얇은 슬랙스가 잘 어울립니다."
            ],
            "warm": [
                f"{age_group} {gender}님께는 스트라이프 셔츠와 치노 팬츠를 추천합니다.",
                f"{age_group} {gender}님은 베이지 자켓과 진청 데님 조합이 좋습니다.",
                f"{age_group} {gender}님께는 오버핏 니트와 슬랙스가 세련됩니다."
            ],
            "cool": [
                f"{age_group} {gender}님은 가죽 자켓과 블랙 스키니진이 잘 어울립니다.",
                f"{age_group} {gender}님께는 코듀로이 자켓과 데님 팬츠를 추천합니다.",
                f"{age_group} {gender}님은 트렌치코트와 니트 원피스가 좋습니다."
            ],
            "cold": [
                f"{age_group} {gender}님께는 패딩과 머플러 조합이 따뜻합니다.",
                f"{age_group} {gender}님은 더플코트와 기모 슬랙스를 추천합니다.",
                f"{age_group} {gender}님은 롱패딩과 청바지가 무난합니다."
            ]
        },
        "practical": {
            "hot": [
                f"{age_group} {gender}님은 면 셔츠와 반바지 조합이 시원하고 편합니다.",
                f"{age_group} {gender}님께는 통풍이 잘 되는 린넨 셔츠와 면바지를 추천합니다."
            ],
            "warm": [
                f"{age_group} {gender}님은 가벼운 점퍼와 슬랙스가 좋습니다.",
                f"{age_group} {gender}님께는 면 니트와 치노 팬츠가 무난합니다."
            ],
            "cool": [
                f"{age_group} {gender}님은 울 자켓과 면바지를 추천합니다.",
                f"{age_group} {gender}님께는 방수 점퍼와 데님이 좋습니다."
            ],
            "cold": [
                f"{age_group} {gender}님은 두꺼운 패딩과 기모 바지를 추천합니다.",
                f"{age_group} {gender}님께는 누빔 코트와 울 바지가 따뜻합니다."
            ]
        },
        "luxury": {
            "hot": [
                f"{age_group} {gender}님은 실크 셔츠와 린넨 팬츠로 고급스러운 여름 스타일을 완성하세요.",
                f"{age_group} {gender}님께는 화이트 원피스와 금빛 샌들이 잘 어울립니다."
            ],
            "warm": [
                f"{age_group} {gender}님은 카디건과 플리츠 스커트를 추천합니다.",
                f"{age_group} {gender}님께는 블레이저와 슬림핏 팬츠가 세련됩니다."
            ],
            "cool": [
                f"{age_group} {gender}님은 캐시미어 코트와 가죽 부츠가 잘 어울립니다.",
                f"{age_group} {gender}님께는 울 재킷과 와이드 슬랙스를 추천합니다."
            ],
            "cold": [
                f"{age_group} {gender}님은 퍼 코트와 힐 조합이 좋습니다.",
                f"{age_group} {gender}님께는 고급 패딩과 가죽 장갑을 추천합니다."
            ]
        },
        "cute": {
            "hot": [
                f"{age_group} {gender}님은 파스텔 티셔츠와 플레어 스커트를 추천합니다.",
                f"{age_group} {gender}님께는 민소매 블라우스와 데님 스커트가 잘 어울립니다."
            ],
            "warm": [
                f"{age_group} {gender}님은 가디건과 하이웨스트 팬츠가 귀엽습니다.",
                f"{age_group} {gender}님께는 셔츠원피스와 플랫슈즈를 추천합니다."
            ],
            "cool": [
                f"{age_group} {gender}님은 체크 코트와 니트 스커트가 잘 어울립니다.",
                f"{age_group} {gender}님께는 후드집업과 데님 팬츠를 추천합니다."
            ],
            "cold": [
                f"{age_group} {gender}님은 더플코트와 니트 비니가 귀엽습니다.",
                f"{age_group} {gender}님께는 패딩과 레깅스 조합을 추천합니다."
            ]
        },
        "gentle": {
            "hot": [
                f"{age_group} {gender}님께는 린넨 셔츠와 슬랙스, 로퍼가 잘 어울립니다.",
                f"{age_group} {gender}님은 화이트 셔츠와 베이지 치노팬츠를 추천합니다."
            ],
            "warm": [
                f"{age_group} {gender}님은 네이비 블레이저와 그레이 슬랙스가 멋집니다.",
                f"{age_group} {gender}님께는 카디건과 면바지가 무난합니다."
            ],
            "cool": [
                f"{age_group} {gender}님은 울 코트와 더비슈즈가 잘 어울립니다.",
                f"{age_group} {gender}님께는 가죽 자켓과 슬랙스를 추천합니다."
            ],
            "cold": [
                f"{age_group} {gender}님은 더블 코트와 머플러 조합이 좋습니다.",
                f"{age_group} {gender}님께는 패딩과 구두를 추천합니다."
            ]
        }
    }

    if temp >= 27:
        return random.choice(outfits[character]["hot"])
    elif temp >= 20:
        return random.choice(outfits[character]["warm"])
    elif temp >= 10:
        return random.choice(outfits[character]["cool"])
    else:
        return random.choice(outfits[character]["cold"])

@app.route("/")
def select_character():
    return render_template("chat_select.html", characters=characters)

@app.route("/start_chat", methods=["POST"])
def start_chat():
    session["character"] = request.form.get("character")
    session["step"] = "ask_city"
    return render_template("chat.html", character=characters[session["character"]])

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()
    step = session.get("step", "ask_city")

    if step == "ask_city":
        session["city"] = user_msg
        session["step"] = "ask_gender"
        return jsonify({"reply": "성별을 입력해주세요 (남성/여성)"})

    elif step == "ask_gender":
        session["gender"] = user_msg
        session["step"] = "ask_age"
        return jsonify({"reply": "나이를 입력해주세요 (숫자)"})

    elif step == "ask_age":
        try:
            age = int(user_msg)
        except ValueError:
            return jsonify({"reply": "나이는 숫자로 입력해주세요."})
        session["age"] = age
        temp, desc = get_weather(session["city"])
        if temp is None:
            return jsonify({"reply": "지원하지 않는 지역입니다. 다시 입력해주세요."})
        outfit = get_outfit(session["character"], session["gender"], age, temp)
        session["step"] = "done"
        return jsonify({"reply": f"{session['city']}의 현재 기온은 {temp}°C, 날씨는 '{desc}'입니다.\n오늘의 추천 코디: {outfit}"})

    elif step == "done":
        return jsonify({"reply": "대화를 다시 시작하려면 페이지를 새로고침 해주세요."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render 포트 사용
    app.run(host="0.0.0.0", port=port)

