from flask import Flask, render_template, request, session, jsonify
import os, json, requests

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

# -----------------------------
# 환경변수 API 키
# -----------------------------
API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# -----------------------------
# 캐릭터 레이블
# -----------------------------
CHAR_LABEL_KO = {
    "trendy": "트렌디 스타일러",
    "gentle": "신사 스타일러",
    "cute": "큐트 스타일러",
    "practical": "프랙티컬 스타일러",
    "luxury": "럭셔리 스타일러"
}

GENDER_KO = {"male": "남성", "female": "여성"}

# -----------------------------
# 도시 매핑
# -----------------------------
KOR2ENG = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu",
    "인천": "Incheon", "광주": "Gwangju", "대전": "Daejeon"
}

# -----------------------------
# JSON 불러오기 (모든 조합 포함)
# -----------------------------
with open("outfits.json", encoding="utf-8") as f:
    OUTFITS = json.load(f)

# -----------------------------
# 유틸 함수
# -----------------------------
def get_age_group(age):
    if age < 20: return "10대"
    elif age < 30: return "20대"
    elif age < 40: return "30대"
    else: return "40대 이상"

def get_weather_group(temp):
    if temp >= 25: return "더움"
    elif temp >= 15: return "선선"
    else: return "추움"

def fetch_weather(city_kor):
    city_eng = KOR2ENG.get(city_kor)
    if not city_eng or not API_KEY:
        return None, None, None
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_eng}&appid={API_KEY}&units=metric&lang=kr"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        temp = float(data["main"]["temp"])
        desc = data["weather"][0]["description"]
        return city_kor, temp, desc
    except:
        return None, None, None

# -----------------------------
# 라우팅
# -----------------------------
@app.route("/")
def select_character():
    return render_template("chat_select.html", characters=CHAR_LABEL_KO)

@app.route("/start", methods=["POST"])
def start_chat():
    character = request.form.get("character")
    session.clear()
    session.update({
        "character": character,
        "step": "gender",
        "messages": [
            {"sender": "bot", "text": f"{CHAR_LABEL_KO.get(character, character)}를 선택하셨어요. 성별을 입력해주세요 (남/여)."}
        ]
    })
    return render_template("chat.html", messages=session["messages"])

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.form.get("message","").strip()
    step = session.get("step")
    character = session.get("character")
    messages = session.get("messages", [])
    if user_msg: messages.append({"sender": "user", "text": user_msg})

    # -----------------------------
    # 1) 성별
    # -----------------------------
    if step == "gender":
        if "남" in user_msg: session["gender"]="male"
        elif "여" in user_msg: session["gender"]="female"
        else:
            messages.append({"sender":"bot","text":"성별은 '남' 또는 '여'로 입력해주세요."})
            session["messages"]=messages
            return jsonify({"messages":messages})
        session["step"]="age"
        messages.append({"sender":"bot","text":"나이를 숫자로 입력해주세요."})

    # -----------------------------
    # 2) 나이
    # -----------------------------
    elif step=="age":
        try:
            age=int(user_msg)
            session["age"]=age
            session["step"]="city"
            messages.append({"sender":"bot","text":"지역을 입력해주세요 (예: 서울, 부산, 대구)"})
        except:
            messages.append({"sender":"bot","text":"나이는 숫자로 입력해주세요."})

    # -----------------------------
    # 3) 지역
    # -----------------------------
    elif step=="city":
        session["city"]=user_msg
        session["step"]="body_shape"
        messages.append({"sender":"bot","text":"체형을 입력해주세요 (마른, 보통, 통통, 역삼각형)"})

    # -----------------------------
    # 4) 체형
    # -----------------------------
    elif step=="body_shape":
        session["body_shape"]=user_msg
        session["step"]="skin_tone"
        messages.append({"sender":"bot","text":"피부 톤을 입력해주세요 (웜톤, 쿨톤)"})

    # -----------------------------
    # 5) 피부톤 -> 최종 코디
    # -----------------------------
    elif step=="skin_tone":
        session["skin_tone"]=user_msg
        city,temp,desc=fetch_weather(session.get("city",""))
        if not city:
            messages.append({"sender":"bot","text":"날씨 정보를 불러오지 못했어요. 도시명을 확인해주세요."})
            session["step"]="city"
            session["messages"]=messages
            return jsonify({"messages":messages})

        age_group = get_age_group(session["age"])
        weather_group = get_weather_group(temp)
        gender=session["gender"]
        skin_tone=session["skin_tone"]
        body_shape=session["body_shape"]

        # JSON에서 코디 가져오기
        try:
            items = OUTFITS[character][gender][age_group][weather_group][skin_tone][body_shape]
            final_text=f"{city} · {temp:.1f}°C · {desc}\n{age_group} {GENDER_KO[gender]} · {weather_group} · 컨셉 ({CHAR_LABEL_KO[character]}) · {skin_tone} · 체형: {body_shape}\n\n상의: {items['상의']}\n하의: {items['하의']}\n신발: {items['신발']}\n액세서리: {items['액세서리']}"
        except:
            final_text="해당 조건의 코디 데이터가 없습니다. 다른 조건으로 시도해보세요."

        messages.append({"sender":"bot","text":final_text})
        session["step"]="done"

    else:
        messages.append({"sender":"bot","text":"대화를 다시 시작하려면 새로고침(F5) 하세요."})

    session["messages"]=messages
    return jsonify({"messages":messages})

# -----------------------------
# 실행
# -----------------------------
if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
