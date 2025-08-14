from flask import Flask, render_template, request, session, jsonify, redirect, url_for
import requests
import os
import random

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

# 한글 → 영어 도시 변환
korean_to_english = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu",
    "인천": "Incheon", "광주": "Gwangju", "대전": "Daejeon",
    "울산": "Ulsan", "세종": "Sejong", "제주": "Jeju",
    "경기": "Gyeonggi-do", "강원": "Gangwon-do", "충북": "Chungcheongbuk-do",
    "충남": "Chungcheongnam-do", "전북": "Jeollabuk-do", "전남": "Jeollanam-do",
    "경북": "Gyeongsangbuk-do", "경남": "Gyeongsangnam-do"
}

# 나이대
def get_age_group(age):
    try:
        age = int(age)
    except:
        return None
    if age < 20: return "10대"
    if age < 30: return "20대"
    if age < 40: return "30대"
    return "40대 이상"

# 캐릭터 정의
PERSONAS = {
    "trendy": "트렌디 전문가",
    "practical": "실속파 코디 장인",
    "luxury": "럭셔리 스타일리스트",
    "cute": "귀여운 패션 친구",
    "gentle": "신사 스타일 챗봇",  # 🕴
}

def normalize_gender(g):
    g = (g or "").strip()
    if g.startswith("남"): return "남성"
    if g.startswith("여"): return "여성"
    return g or "중립"

# 캐릭터별 문체
def style_tone(persona, text_lines):
    text = "\n".join(text_lines)
    if persona == "trendy":
        return f"[트렌디] 요즘 감성으로 이렇게 가보시죠!\n{text}"
    if persona == "practical":
        return f"[실속파] 옷장에 있는 걸로 충분히 멋낼 수 있어요:\n{text}"
    if persona == "luxury":
        return f"[럭셔리] 고급스러운 무드로 제안드립니다:\n{text}"
    if persona == "cute":
        return f"[귀여운] 이렇게 입으면 찰떡일 듯..! ✨\n{text}"
    if persona == "gentle":
        return f"[신사] 품격 있게 제안드립니다.\n{text}"
    return text

# 캐릭터 + 성별 + 나이대 + 기온 기반 코디 후보 생성
def build_outfit_list(temp, gender, age_group, persona):
    gender = normalize_gender(gender)
    s = []

    # 기본 베이스 (성별/기온/나이대)
    if gender == "남성":
        if temp >= 27:
            s += [
                "오버핏 반팔 셔츠 + 코튼 반바지 + 스니커즈",
                "린넨 반팔 + 베이지 쇼츠 + 샌들",
                "드라이핏 티 + 조거 반바지 + 캡모자"
            ]
        elif temp >= 20:
            s += [
                "얇은 셔츠 + 슬랙스 + 로퍼",
                "맨투맨 + 와이드 팬츠 + 스니커즈",
                "가디건 + 코튼팬츠 + 캔버스화"
            ]
        elif temp >= 10:
            s += [
                "니트 + 데님 + 첼시부츠",
                "블루종 + 치노 + 로퍼",
                "트렌치코트 + 셔츠 + 슬랙스"
            ]
        else:
            s += [
                "롱패딩 + 터틀넥 + 기모 슬랙스 + 머플러",
                "울 코트 + 니트 + 데님 + 부츠"
            ]
    elif gender == "여성":
        if temp >= 27:
            s += [
                "민소매 원피스 + 샌들 + 버킷햇",
                "크롭티 + 하이웨스트 반바지 + 스니커즈",
                "린넨 셔츠 + 플레어 스커트 + 플랫"
            ]
        elif temp >= 20:
            s += [
                "셔츠 원피스 + 스니커즈",
                "블라우스 + 슬랙스 + 로퍼",
                "가디건 + 데님 스커트 + 캔버스"
            ]
        elif temp >= 10:
            s += [
                "트렌치코트 + 니트 + 슬랙스",
                "가디건 + 플리츠 스커트 + 플랫슈즈",
                "블루종 + 데님 + 앵클부츠"
            ]
        else:
            s += [
                "롱코트 + 터틀넥 + 울 팬츠 + 부츠",
                "숏패딩 + 기모 원피스 + 롱부츠"
            ]
    else:
        # 중립 기본
        if temp >= 20:
            s += ["티셔츠 + 면바지 + 스니커즈", "셔츠 + 슬랙스 + 로퍼"]
        else:
            s += ["니트 + 코트 + 데님 + 부츠", "패딩 + 기모팬츠 + 방한화"]

    # 나이대 보정
    if age_group in ("30대", "40대 이상"):
        s += ["미니멀 셔츠 + 테이퍼드 슬랙스 + 레더슈즈", "울 니트 + 울 팬츠 + 로퍼"]

    # 캐릭터별 가중치/추가 후보
    if persona == "trendy":
        s += [
            "테크웨어 점퍼 + 카고팬츠 + 러닝슈즈",
            "스트릿 로고 티 + 와이드 데님 + 스케이트화"
        ]
    elif persona == "practical":
        s += [
            "기본 라운드 티 + 치노팬츠 + 캔버스화",
            "베이식 셔츠 + 일자 데님 + 스니커즈"
        ]
    elif persona == "luxury":
        s += [
            "캐시미어 니트 + 울 슬랙스 + 페니 로퍼",
            "실크 블라우스 + 테일러드 팬츠 + 힐/더비"
        ]
    elif persona == "cute":
        s += [
            "파스텔 가디건 + 미니스커트(또는 쇼츠) + 메리제인/로퍼",
            "리본/버클 포인트 블라우스 + 플레어 팬츠 + 플랫 ✨"
        ]
    elif persona == "gentle":
        # 신사 스타일: 포멀/세미포멀 위주
        if temp >= 27:
            s += [
                "시어서커 재킷 + 화이트 셔츠 + 라이트 슬랙스 + 로퍼",
                "린넨 셔츠 + 네이비 치노 + 로퍼"
            ]
        elif temp >= 20:
            s += [
                "네이비 블레이저 + 옥스퍼드 셔츠 + 그레이 슬랙스 + 브라운 로퍼",
                "라이트 가디건 + 버튼다운 셔츠 + 치노 + 더비슈즈"
            ]
        elif temp >= 10:
            s += [
                "트렌치코트 + 니트 타이 + 슬랙스 + 첼시부츠",
                "울 재킷 + 터틀넥 + 플란넬 슬랙스 + 로퍼"
            ]
        else:
            s += [
                "캐시미어 코트 + 머플러 + 기모 슬랙스 + 레더 부츠",
                "다운 코트 + 니트 + 울 팬츠 + 글러브"
            ]

    # 중복 제거 + 랜덤 3개
    s = list(dict.fromkeys(s))
    return random.sample(s, min(3, len(s))) if s else ["기본적인 편안한 코디를 추천합니다."]

@app.route("/", methods=["GET"])
def select():
    # 캐릭터 선택 화면
    return render_template("chat_select.html", personas=PERSONAS)

@app.route("/start", methods=["POST"])
def start():
    persona = request.form.get("persona")
    if persona not in PERSONAS:
        return redirect(url_for("select"))
    session.clear()
    session["persona"] = persona
    session["step"] = 1
    return render_template("chat.html", persona_name=PERSONAS[persona])

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_msg = (data.get("message") or "").strip()
    step = session.get("step", 1)
    persona = session.get("persona", "practical")

    if step == 1:
        session["city"] = user_msg
        session["step"] = 2
        return jsonify({"reply": style_tone(persona, ["성별을 입력해 주세요. (남성 / 여성)"])})

    if step == 2:
        session["gender"] = user_msg
        session["step"] = 3
        return jsonify({"reply": style_tone(persona, ["나이를 숫자로 입력해 주세요. (예: 25)"])})

    if step == 3:
        if not user_msg.isdigit():
            return jsonify({"reply": style_tone(persona, ["나이는 숫자만 입력해 주세요."])})
        session["age"] = int(user_msg)
        session["step"] = 4

        city_kor = session.get("city", "")
        city_eng = korean_to_english.get(city_kor)
        if not city_eng:
            return jsonify({"reply": style_tone(persona, [f"'{city_kor}'는 지원하지 않는 지역입니다. 다른 지역을 입력해 주세요."])})

        if not API_KEY:
            return jsonify({"reply": style_tone(persona, ["서버에 OPENWEATHER_API_KEY가 설정되지 않았습니다. 관리자에게 문의하세요."])})

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_eng}&appid={API_KEY}&units=metric&lang=kr"
        try:
            res = requests.get(url, timeout=8)
        except Exception as e:
            return jsonify({"reply": style_tone(persona, [f"날씨 API 호출 중 오류가 발생했습니다: {e}"])})

        if res.status_code != 200:
            return jsonify({"reply": style_tone(persona, ["날씨 정보를 가져오지 못했습니다. API 키/도시명을 확인해 주세요."])})

        data = res.json()
        temp = data.get("main", {}).get("temp")
        desc = data.get("weather", [{}])[0].get("description", "")
        age_group = get_age_group(session["age"])

        outfits = build_outfit_list(temp, session["gender"], age_group, persona)
        outfit_lines = [f"- {o}" for o in outfits]

        header = [
            f"{city_kor}의 현재 온도는 {temp}°C, 날씨는 '{desc}'입니다.",
            f"{age_group} {normalize_gender(session['gender'])}을(를) 위한 추천 코디:"
        ]
        reply = style_tone(persona, header + outfit_lines)
        return jsonify({"reply": reply})

    return jsonify({"reply": style_tone(persona, ["대화를 다시 시작하려면 새로고침 해 주세요."]) })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
