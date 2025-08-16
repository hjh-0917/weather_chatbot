from flask import Flask, render_template, request, session, jsonify
import os
import requests

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

# -----------------------------
# OpenWeather API
# -----------------------------
API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# 한글 → 영문 도시명 (대표 도시 위주)
KOR2ENG = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu", "인천": "Incheon", "광주": "Gwangju",
    "대전": "Daejeon", "울산": "Ulsan", "세종": "Sejong", "수원": "Suwon", "창원": "Changwon",
    "춘천": "Chuncheon", "청주": "Cheongju", "전주": "Jeonju", "포항": "Pohang", "제주": "Jeju",
    "김해": "Gimhae", "용인": "Yongin", "성남": "Seongnam", "안양": "Anyang", "의정부": "Uijeongbu"
}

GENDER_KO = {"male": "남성", "female": "여성"}
CHAR_LABEL_KO = {
    "trendy": "트렌디 스타일러",
    "gentle": "신사 스타일러",
    "cute": "큐트 스타일러",
    "practical": "프랙티컬 스타일러",
    "luxury": "럭셔리 스타일러",
}

# -----------------------------
# 유틸
# -----------------------------
def get_age_group(age: int) -> str:
    if age < 20:
        return "10대"
    elif age < 30:
        return "20대"
    elif age < 40:
        return "30대"
    else:
        return "40대 이상"

def get_weather_group(temp_c: float) -> str:
    if temp_c >= 25:
        return "더움"
    elif temp_c >= 15:
        return "선선"
    else:
        return "추움"

def fetch_weather(city_kor: str):
    """OpenWeather에서 현재 기온/날씨(한글)를 받아온다."""
    city_eng = KOR2ENG.get(city_kor)
    if not city_eng:
        return None, None, None  # 매핑 실패
    if not API_KEY:
        return None, None, None  # API 키 없음
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city_eng}&appid={API_KEY}&units=metric&lang=kr"
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
        if r.status_code == 200 and "main" in data and "weather" in data:
            temp = float(data["main"]["temp"])
            desc = data["weather"][0]["description"]
            return city_kor, temp, desc
        return None, None, None
    except Exception:
        return None, None, None

# -----------------------------
# 베이스 코디: 캐릭터 × 날씨 × 성별
# (나이/체형/피부톤은 아래에서 가공)
# -----------------------------
BASE_OUTFITS = {
    "trendy": {
        "더움": {
            "male":   {"상의": "오버핏 반팔 티셔츠", "하의": "와이드 데님", "신발": "캔버스 스니커즈", "액세서리": "버킷햇"},
            "female": {"상의": "크롭 티셔츠",       "하의": "하이웨스트 데님 쇼츠", "신발": "스트랩 샌들", "액세서리": "크로스백"},
        },
        "선선": {
            "male":   {"상의": "오버셔츠", "하의": "카고 팬츠", "신발": "로우탑 스니커즈", "액세서리": "캡"},
            "female": {"상의": "카라 가디건", "하의": "플리츠 스커트", "신발": "스니커즈", "액세서리": "토트백"},
        },
        "추움": {
            "male":   {"상의": "플리스 후디", "하의": "조거 팬츠", "신발": "하이탑 스니커즈", "액세서리": "비니"},
            "female": {"상의": "숏 패딩", "하의": "스트레이트 데님", "신발": "앵클 부츠", "액세서리": "버킷백"},
        },
    },
    "gentle": {
        "더움": {
            "male":   {"상의": "린넨 셔츠", "하의": "테이퍼드 치노", "신발": "로퍼", "액세서리": "가죽 벨트"},
            "female": {"상의": "린넨 블라우스", "하의": "미디 스커트", "신발": "플랫 슈즈", "액세서리": "가죽 벨트"},
        },
        "선선": {
            "male":   {"상의": "니트 폴로", "하의": "슬랙스", "신발": "더비 슈즈", "액세서리": "메탈 워치"},
            "female": {"상의": "크롭 가디건", "하의": "슬랙스", "신발": "로퍼", "액세서리": "실버 네클리스"},
        },
        "추움": {
            "male":   {"상의": "울 코트 + 터틀넥", "하의": "울 슬랙스", "신발": "첼시 부츠", "액세서리": "레더 글러브"},
            "female": {"상의": "핸드메이드 코트", "하의": "니트 원피스", "신발": "롱 부츠", "액세서리": "레더 토트"},
        },
    },
    "cute": {
        "더움": {
            "male":   {"상의": "파스텔 티셔츠", "하의": "스트레이트 데님", "신발": "화이트 스니커즈", "액세서리": "캔버스 백"},
            "female": {"상의": "퍼프 블라우스", "하의": "A라인 스커트", "신발": "메리제인", "액세서리": "헤어핀"},
        },
        "선선": {
            "male":   {"상의": "라이트 가디건", "하의": "크롭트 치노", "신발": "스니커즈", "액세서리": "비니"},
            "female": {"상의": "리본 블라우스", "하의": "플리츠 스커트", "신발": "플랫 슈즈", "액세서리": "리본 헤어밴드"},
        },
        "추움": {
            "male":   {"상의": "더플 코트 + 니트", "하의": "면 팬츠", "신발": "스니커즈", "액세서리": "머플러"},
            "female": {"상의": "더플 코트 + 케이블 니트", "하의": "미니 스커트", "신발": "롱 부츠", "액세서리": "머플러"},
        },
    },
    "practical": {
        "더움": {
            "male":   {"상의": "기능성 티셔츠", "하의": "카고 쇼츠", "신발": "트레일 러너", "액세서리": "캡"},
            "female": {"상의": "드라이 티셔츠", "하의": "테크 쇼츠", "신발": "워킹 슈즈", "액세서리": "캡"},
        },
        "선선": {
            "male":   {"상의": "바람막이", "하의": "기능성 팬츠", "신발": "워킹 슈즈", "액세서리": "크로스백"},
            "female": {"상의": "라이트 재킷", "하의": "조거 팬츠", "신발": "스니커즈", "액세서리": "백팩"},
        },
        "추움": {
            "male":   {"상의": "소프트쉘 재킷", "하의": "기모 조거", "신발": "고어텍스 부츠", "액세서리": "니트 비니"},
            "female": {"상의": "패딩 재킷", "하의": "기모 레깅스", "신발": "워킹 부츠", "액세서리": "니트 비니"},
        },
    },
    "luxury": {
        "더움": {
            "male":   {"상의": "실크 셔츠", "하의": "플리츠 슬랙스", "신발": "로퍼", "액세서리": "가죽 카드지갑"},
            "female": {"상의": "실크 블라우스", "하의": "크림 슬랙스", "신발": "뮬", "액세서리": "미니 가죽 백"},
        },
        "선선": {
            "male":   {"상의": "캐시미어 가디건", "하의": "울 슬랙스", "신발": "더비 슈즈", "액세서리": "레더 벨트"},
            "female": {"상의": "트위드 자켓", "하의": "플레어 스커트", "신발": "펌프스", "액세서리": "진주 귀걸이"},
        },
        "추움": {
            "male":   {"상의": "체스터필드 코트 + 캐시미어 터틀넥", "하의": "플란넬 슬랙스", "신발": "첼시 부츠", "액세서리": "레더 글러브"},
            "female": {"상의": "롱 코트 + 실크 블라우스", "하의": "와이드 슬랙스", "신발": "첼시 부츠", "액세서리": "레더 글러브"},
        },
    },
}

# -----------------------------
# 피부톤 팔레트 (캐릭터별 추천 색)
# -----------------------------
PALETTES = {
    "trendy": {
        "웜톤": {"상의": "베이지", "하의": "카키", "신발": "화이트", "액세서리": "골드"},
        "쿨톤": {"상의": "네이비", "하의": "그레이", "신발": "블랙", "액세서리": "실버"},
    },
    "gentle": {
        "웜톤": {"상의": "아이보리", "하의": "카멜", "신발": "브라운", "액세서리": "골드"},
        "쿨톤": {"상의": "라이트 블루", "하의": "차콜", "신발": "블랙", "액세서리": "실버"},
    },
    "cute": {
        "웜톤": {"상의": "코랄", "하의": "크림", "신발": "화이트", "액세서리": "골드"},
        "쿨톤": {"상의": "라벤더", "하의": "아이보리", "신발": "화이트", "액세서리": "실버"},
    },
    "practical": {
        "웜톤": {"상의": "올리브", "하의": "샌드", "신발": "브라운", "액세서리": "카키"},
        "쿨톤": {"상의": "스카이블루", "하의": "그레이", "신발": "블랙", "액세서리": "네이비"},
    },
    "luxury": {
        "웜톤": {"상의": "카멜", "하의": "초콜릿", "신발": "버건디", "액세서리": "골드"},
        "쿨톤": {"상의": "차콜", "하의": "네이비", "신발": "블랙", "액세서리": "실버"},
    },
}

# -----------------------------
# 가공 함수
# -----------------------------
def deep_copy_items(items: dict) -> dict:
    return {k: v for k, v in items.items()}

def tune_by_age(character: str, gender: str, age_group: str, items: dict) -> dict:
    """나이대에 따라 살짝 톤앤매너를 조정 (캐릭터 성격 유지)."""
    it = deep_copy_items(items)

    if age_group == "10대":
        if character in ["gentle", "luxury"]:
            # 포멀도를 살짝 낮춤
            if gender == "male":
                it["신발"] = "화이트 스니커즈"
            else:
                it["신발"] = "플랫 슈즈"
        if character == "trendy" and "슬랙스" in it["하의"]:
            it["하의"] = "와이드 데님"

    elif age_group == "30대":
        if character == "trendy" and "와이드 데님" in it["하의"]:
            it["하의"] = "세미와이드 슬랙스"
        if character == "cute" and "미니 스커트" in it["하의"]:
            it["하의"] = "A라인 미디 스커트"

    elif age_group == "40대 이상":
        # 전체적으로 실루엣 정돈
        if "오버핏" in it["상의"]:
            it["상의"] = it["상의"].replace("오버핏", "세미오버핏")
        if "와이드" in it["하의"]:
            it["하의"] = it["하의"].replace("와이드", "세미와이드")
        if character in ["trendy", "cute"]:
            it["신발"] = "레더 로퍼" if gender == "male" else "로우 힐"

    return it

def adjust_by_body_shape(items: dict, body_shape: str, gender: str) -> dict:
    """체형별로 핏/실루엣 보정."""
    it = deep_copy_items(items)
    b = body_shape.strip()

    def prefix_fit(text, fit_word):
        # 이미 존재하면 중복 방지
        return text if text.startswith(fit_word) else f"{fit_word} {text}"

    if b == "마른":
        it["상의"] = prefix_fit(it["상의"], "오버핏")
        if "슬랙스" in it["하의"]:
            it["하의"] = it["하의"].replace("슬랙스", "스트레이트 슬랙스")
        elif "데님" in it["하의"]:
            it["하의"] = it["하의"].replace("데님", "스트레이트 데님")

    elif b == "통통":
        it["상의"] = prefix_fit(it["상의"], "세미오버핏")
        # 하의는 테이퍼드/다크 톤 지향
        if "와이드" in it["하의"]:
            it["하의"] = it["하의"].replace("와이드", "테이퍼드")
        if "데님" in it["하의"] and "다크" not in it["하의"]:
            it["하의"] = "다크 데님" if gender == "male" else "다크 스트레이트 데님"

    elif b == "역삼각형":
        # 상체는 정돈, 하체에 볼륨
        it["상의"] = prefix_fit(it["상의"], "레귤러핏")
        if "슬랙스" in it["하의"] and "와이드" not in it["하의"]:
            it["하의"] = it["하의"].replace("슬랙스", "와이드 슬랙스")
        if "스커트" in it["하의"] and "플리츠" not in it["하의"]:
            it["하의"] = "플리츠 스커트"

    # 보통: 변화 없음
    return it

def colorize_by_skin_tone(character: str, skin_tone: str, items: dict) -> dict:
    """피부톤에 따라 카테고리별 대표 색을 붙여 최종 아이템 문구를 만든다."""
    palette = PALETTES.get(character, {}).get(skin_tone, {})
    it = deep_copy_items(items)
    for cat in ["상의", "하의", "신발", "액세서리"]:
        color = palette.get(cat)
        if color:
            it[cat] = f"{color} {it[cat]}"
    return it

def build_final_text(city, temp_c, desc, age_group, gender, weather_group, character, skin_tone, body_shape, items):
    """요청한 최종 출력 포맷으로 문자열 생성."""
    gender_ko = GENDER_KO.get(gender, "")
    concept = CHAR_LABEL_KO.get(character, character)
    header = f"{city} · {temp_c:.1f}°C · '{desc}'\n{age_group} {gender_ko} · {weather_group} · 컨셉 ({concept}) · {skin_tone} · 체형: {body_shape}"
    detail = f"\n\n상의: {items['상의']}\n하의: {items['하의']}\n신발: {items['신발']}\n액세서리: {items['액세서리']}"
    return header + detail

# -----------------------------
# 라우팅 (대화 흐름)
# -----------------------------
@app.route("/")
def select_character():
    return render_template("chat_select.html")

@app.route("/start", methods=["POST"])
def start_chat():
    character = request.form.get("character")
    session.clear()
    session["character"] = character
    session["step"] = "gender"
    session["messages"] = [
        {"sender": "bot", "text": f"{CHAR_LABEL_KO.get(character, character)}를 선택하셨어요. 성별을 입력해주세요 (남/여)."}
    ]
    return render_template("chat.html", messages=session["messages"])

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.form.get("message", "").strip()
    step = session.get("step")
    character = session.get("character")
    messages = session.get("messages", [])
    if user_msg:
        messages.append({"sender": "user", "text": user_msg})

    # 1) 성별
    if step == "gender":
        if "남" in user_msg:
            session["gender"] = "male"
        elif "여" in user_msg:
            session["gender"] = "female"
        else:
            bot = "성별은 '남' 또는 '여'로 입력해주세요."
            messages.append({"sender": "bot", "text": bot})
            session["messages"] = messages
            return jsonify({"messages": messages})

        session["step"] = "age"
        bot = "나이를 숫자로 입력해주세요."
        messages.append({"sender": "bot", "text": bot})

    # 2) 나이
    elif step == "age":
        try:
            age = int(user_msg)
            if age <= 0 or age > 120:
                raise ValueError()
            session["age"] = age
            session["step"] = "city"
            bot = "지역을 입력해주세요 (예: 서울, 부산, 대구, 인천, 광주, 대전, 울산, 세종, 수원, 전주, 제주)"
        except ValueError:
            bot = "나이는 숫자로 입력해주세요 (예: 22)."
        messages.append({"sender": "bot", "text": bot})

    # 3) 지역
    elif step == "city":
        city = user_msg.replace(" ", "")
        session["city"] = city
        # 여기서는 우선 도시만 저장하고 다음 단계로 진행 (실제 조회는 최종 단계에서)
        session["step"] = "body_shape"
        bot = "당신의 체형을 알려주세요\n예시) 마른, 보통, 통통, 역삼각형"
        messages.append({"sender": "bot", "text": bot})

    # 4) 체형
    elif step == "body_shape":
        session["body_shape"] = user_msg
        session["step"] = "skin_tone"
        bot = "당신의 피부 톤을 알려주세요\n예시) 웜톤, 쿨톤"
        messages.append({"sender": "bot", "text": bot})

    # 5) 피부톤 -> 최종 추천
    elif step == "skin_tone":
        session["skin_tone"] = user_msg
        # 날씨 조회
        city, temp, desc = fetch_weather(session.get("city", ""))
        if not city:
            bot = "날씨 정보를 불러오지 못했어요. 지원 도시명과 API 키를 확인하고 다시 '지역'부터 입력해 주세요."
            session["step"] = "city"
            messages.append({"sender": "bot", "text": bot})
            session["messages"] = messages
            return jsonify({"messages": messages})

        # 그룹 계산
        age_group = get_age_group(int(session["age"]))
        weather_group = get_weather_group(temp)
        gender = session["gender"]
        skin_tone = session["skin_tone"]
        body_shape = session["body_shape"]

        # 베이스 아이템
        base = BASE_OUTFITS.get(character, {}).get(weather_group, {}).get(gender, {})
        if not base:
            bot = "해당 조건의 코디 데이터가 부족해요. 다른 캐릭터나 조건으로 시도해 보세요."
            session["step"] = "done"
            messages.append({"sender": "bot", "text": bot})
            session["messages"] = messages
            return jsonify({"messages": messages})

        # 가공: 나이 → 체형 → 피부톤
        items = tune_by_age(character, gender, age_group, base)
        items = adjust_by_body_shape(items, body_shape, gender)
        items = colorize_by_skin_tone(character, skin_tone, items)

        # 최종 텍스트
        final_text = build_final_text(
            city=city, temp_c=temp, desc=desc,
            age_group=age_group, gender=gender,
            weather_group=weather_group, character=character,
            skin_tone=skin_tone, body_shape=body_shape,
            items=items
        )
        bot = final_text
        session["step"] = "done"
        messages.append({"sender": "bot", "text": bot})

    else:
        bot = "대화를 다시 시작하려면 새로고침(F5) 하세요."
        messages.append({"sender": "bot", "text": bot})

    session["messages"] = messages
    return jsonify({"messages": messages})

# -----------------------------
# Render 실행용
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

