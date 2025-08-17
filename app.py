import os
import json
from pathlib import Path
from flask import Flask, render_template, request, session, jsonify
import requests

app = Flask(__name__)

# 세션용 시크릿키
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

# OpenWeatherMap
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "").strip()

# 한글 도시명 → 영문 도시명 (없으면 입력값 그대로 시도)
KOR_TO_ENG_CITY = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu", "인천": "Incheon",
    "광주": "Gwangju", "대전": "Daejeon", "울산": "Ulsan", "세종": "Sejong",
    "경기": "Gyeonggi-do", "강원": "Gangwon-do", "충북": "Chungcheongbuk-do",
    "충남": "Chungcheongnam-do", "전북": "Jeollabuk-do", "전남": "Jeollanam-do",
    "경북": "Gyeongsangbuk-do", "경남": "Gyeongsangnam-do", "제주": "Jeju"
}

# 캐릭터 표기
PERSONA_LABEL = {
    "trendy": "트렌디",
    "practical": "실용파",
    "luxury": "럭셔리",
    "gentle": "신사",
    "cute": "귀여움"
}

# 캐릭터 이미지 파일명
PERSONA_IMAGE = {
    "trendy": "trendy.png",
    "practical": "practical.png",
    "luxury": "luxury.png",
    "gentle": "gentle.png",
    "cute": "cute.png"
}

# JSON 데이터 루트 경로
JSON_ROOT = Path(__file__).parent / "JSON"

# 대화 진행 단계 정의
STEPS = ["location", "gender", "age", "skin_tone", "body_shape"]

def normalize_gender(text: str) -> str:
    t = text.strip().lower()
    if t in ["남", "남자", "male", "m", "boy"]:
        return "male"
    if t in ["여", "여자", "female", "f", "girl"]:
        return "female"
    return ""

def age_to_label_and_dir(age: int, gender: str):
    """
    나이 → JSON 디렉터리/라벨 매핑
    - 10~19 : teen_{gender} / "10대"
    - 20~29 : twenties_{gender} / "20대"
    - 그 외  : 가장 가까운(20대)로 폴백
    """
    if age <= 19:
        return f"teen_{gender}", "10대"
    if 20 <= age <= 29:
        return f"twenties_{gender}", "20대"
    # 폴백 (현재 teen, twenties만 있다고 했으니 20대로 통일)
    return f"twenties_{gender}", "20대"

def temp_to_category(temp_c: float) -> str:
    """기온 → '더움'/'선선'/'추움' 구간화"""
    if temp_c >= 24:
        return "더움"
    if temp_c >= 15:
        return "선선"
    return "추움"

def get_weather(city_kor: str):
    """
    현재 기온/설명 가져오기. 실패시 (None, None) 반환
    """
    # 도시 영문화 (없으면 원문 그대로 시도)
    city_query = KOR_TO_ENG_CITY.get(city_kor.strip(), city_kor.strip())
    if not OPENWEATHER_API_KEY:
        return None, None

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city_query}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr"
        )
        r = requests.get(url, timeout=8)
        data = r.json()
        if r.status_code == 200 and "main" in data:
            temp = float(data["main"]["temp"])
            desc = data["weather"][0]["description"]
            return temp, desc
    except Exception:
        pass
    return None, None

def safe_pick(d: dict, key: str):
    """
    d[key]가 없으면 가능한 첫 번째 값을 폴백으로 반환 (딕셔너리면 첫 키로 내려감)
    """
    if isinstance(d, dict):
        if key in d:
            return d[key]
        # 폴백: 첫 번째 항목
        if d:
            first_key = next(iter(d.keys()))
            return d[first_key]
    return None

def load_outfit(persona: str, gender_norm: str, age_label: str, weather_cat: str, skin: str, body: str):
    """
    JSON 구조: {persona: {gender: {age_label: {weather_cat: {skin: {body: {...}}}}}}}
    키가 없을 때는 단계적으로 폴백.
    """
    # 폴백 키
    default_skin = "웜톤"
    default_body = "보통"

    # 파일 경로 결정
    # 디렉터리는 뷰 함수에서 미리 계산해 넘겨주므로 여기선 전달된 age_label만 사용
    # 실제 파일은 persona.json (예: trendy.json)
    return {
        "resolver": lambda p: p  # 자리표시자 (아래에서 교체)
    }

def pick_from_json(json_path: Path, persona: str, gender_norm: str, age_label: str,
                   weather_cat: str, skin: str, body: str):
    """
    실제 JSON에서 탐색 + 폴백
    """
    default_skin = "웜톤"
    default_body = "보통"

    if not json_path.exists():
        return None

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    # 단계별 안전 탐색
    node = safe_pick(data, persona)
    node = safe_pick(node, gender_norm)
    node = safe_pick(node, age_label)
    node = safe_pick(node, weather_cat)

    # 스킨 톤
    node_skin = safe_pick(node, skin)
    if node_skin is None:
        node_skin = safe_pick(node, default_skin)

    # 체형
    outfit = safe_pick(node_skin, body)
    if outfit is None:
        outfit = safe_pick(node_skin, default_body)

    # outfit 예: {"상의": "...", "하의": "...", "신발": "...", "액세서리": "..."}
    if isinstance(outfit, dict):
        return outfit
    return None

@app.route("/", methods=["GET"])
def select():
    return render_template(
        "chat_select.html",
        persona_label=PERSONA_LABEL,
        persona_image=PERSONA_IMAGE
    )

@app.route("/start", methods=["POST"])
def start():
    persona = request.form.get("persona", "").strip()
    if persona not in PERSONA_LABEL:
        return jsonify({"ok": False, "error": "캐릭터 선택이 올바르지 않습니다."}), 400

    session.clear()
    session["persona"] = persona
    session["step"] = 0
    session["answers"] = {}

    return jsonify({"ok": True})

@app.route("/chat", methods=["GET"])
def chat_page():
    persona = session.get("persona", "")
    label = PERSONA_LABEL.get(persona, persona)
    image = PERSONA_IMAGE.get(persona, "")
    return render_template("chat.html", persona=persona, persona_label=label, persona_image=image)

@app.route("/chat", methods=["POST"])
def chat_api():
    # 현재 단계
    step_idx = session.get("step", 0)
    answers = session.get("answers", {})
    persona = session.get("persona", "")

    user_msg = (request.json or {}).get("message", "").strip()

    # 단계 처리
    if step_idx == 0:  # 지역
        if not user_msg:
            return jsonify({"reply": "어느 지역에 계신가요? (예: 서울, 부산)"})
        answers["location"] = user_msg
        session["step"] = 1
        session["answers"] = answers
        return jsonify({"reply": "성별을 알려주세요. (남/여)"})

    if step_idx == 1:  # 성별
        g = normalize_gender(user_msg)
        if not g:
            return jsonify({"reply": "성별을 남/여 로 입력해주세요."})
        answers["gender"] = g
        session["step"] = 2
        session["answers"] = answers
        return jsonify({"reply": "나이를 숫자로 입력해주세요. (예: 17, 22)"})

    if step_idx == 2:  # 나이
        try:
            age = int(user_msg)
            if age <= 0 or age > 120:
                raise ValueError
        except Exception:
            return jsonify({"reply": "나이는 1~120 사이의 숫자로 입력해주세요."})
        answers["age"] = age
        session["step"] = 3
        session["answers"] = answers
        return jsonify({"reply": "피부 톤을 알려주세요. (웜톤/쿨톤)"})

    if step_idx == 3:  # 피부톤
        tone = user_msg.replace("톤", "").strip()
        if tone not in ["웜", "쿨", "웜톤", "쿨톤"]:
            return jsonify({"reply": "피부 톤은 웜톤/쿨톤 중에서 선택해주세요."})
        answers["skin_tone"] = "웜톤" if "웜" in tone else "쿨톤"
        session["step"] = 4
        session["answers"] = answers
        return jsonify({"reply": "체형을 알려주세요. (마른/보통/통통/역삼각형)"})

    if step_idx == 4:  # 체형 → 추천
        if user_msg not in ["마른", "보통", "통통", "역삼각형"]:
            return jsonify({"reply": "체형은 마른/보통/통통/역삼각형 중에서 입력해주세요."})
        answers["body_shape"] = user_msg

        # 날씨 조회
        temp, desc = get_weather(answers["location"])
        # 날씨 카테고리
        if temp is not None:
            weather_cat = temp_to_category(temp)
        else:
            # API 키 없음, 실패 등 폴백
            weather_cat = "선선"
            desc = "날씨 조회 실패(임시로 선선 적용)"

        # 나이/디렉터리/라벨
        age_dir, age_label = age_to_label_and_dir(int(answers["age"]), answers["gender"])

        # json 파일 경로
        json_file = JSON_ROOT / age_dir / f"{persona}.json"
        outfit = pick_from_json(
            json_file,
            persona=persona,
            gender_norm=answers["gender"],
            age_label=age_label,
            weather_cat=weather_cat,
            skin=answers["skin_tone"],
            body=answers["body_shape"]
        )

        if not outfit:
            reply = (
                f"선택 조합에 대한 코디 데이터를 찾지 못했습니다. "
                f"(캐릭터: {PERSONA_LABEL.get(persona, persona)}, 성별: {answers['gender']}, "
                f"나이: {answers['age']}, 날씨: {weather_cat}, 피부톤: {answers['skin_tone']}, 체형: {answers['body_shape']})\n"
                f"JSON 파일을 확인해주세요."
            )
        else:
            # 결과 문자열 구성
            # 도시 표기는 사용자가 입력한 그대로
            loc = answers["location"]
            weather_line = f"{loc}의 현재 날씨: {desc if desc else '정보 없음'}"
            outfit_line = (
                f"[오늘의 코디]\n"
                f"- 상의: {outfit.get('상의','')}\n"
                f"- 하의: {outfit.get('하의','')}\n"
                f"- 신발: {outfit.get('신발','')}\n"
                f"- 액세서리: {outfit.get('액세서리','')}"
            )
            reply = (
                f"{weather_line}\n"
                f"(분류: {weather_cat}, 나이대: {age_label}, 피부톤: {answers['skin_tone']}, 체형: {answers['body_shape']})\n\n"
                f"{outfit_line}"
            )

        # 대화 초기화 옵션 제공
        session["step"] = 0
        session["answers"] = {}
        return jsonify({"reply": reply})

    # 그 외 예외
    session["step"] = 0
    session["answers"] = {}
    return jsonify({"reply": "대화를 다시 시작합니다. 어느 지역에 계신가요? (예: 서울, 부산)"})


# Render 배포용 (개발용 서버도 지원)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
import os
import json
from pathlib import Path
from flask import Flask, render_template, request, session, jsonify
import requests

app = Flask(__name__)

# 세션용 시크릿키
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")

# OpenWeatherMap
OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY", "").strip()

# 한글 도시명 → 영문 도시명 (없으면 입력값 그대로 시도)
KOR_TO_ENG_CITY = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu", "인천": "Incheon",
    "광주": "Gwangju", "대전": "Daejeon", "울산": "Ulsan", "세종": "Sejong",
    "경기": "Gyeonggi-do", "강원": "Gangwon-do", "충북": "Chungcheongbuk-do",
    "충남": "Chungcheongnam-do", "전북": "Jeollabuk-do", "전남": "Jeollanam-do",
    "경북": "Gyeongsangbuk-do", "경남": "Gyeongsangnam-do", "제주": "Jeju"
}

# 캐릭터 표기
PERSONA_LABEL = {
    "trendy": "트렌디",
    "practical": "실용파",
    "luxury": "럭셔리",
    "gentle": "신사",
    "cute": "귀여움"
}

# 캐릭터 이미지 파일명
PERSONA_IMAGE = {
    "trendy": "trendy.png",
    "practical": "practical.png",
    "luxury": "luxury.png",
    "gentle": "gentle.png",
    "cute": "cute.png"
}

# JSON 데이터 루트 경로
JSON_ROOT = Path(__file__).parent / "JSON"

# 대화 진행 단계 정의
STEPS = ["location", "gender", "age", "skin_tone", "body_shape"]

def normalize_gender(text: str) -> str:
    t = text.strip().lower()
    if t in ["남", "남자", "male", "m", "boy"]:
        return "male"
    if t in ["여", "여자", "female", "f", "girl"]:
        return "female"
    return ""

def age_to_label_and_dir(age: int, gender: str):
    """
    나이 → JSON 디렉터리/라벨 매핑
    - 10~19 : teen_{gender} / "10대"
    - 20~29 : twenties_{gender} / "20대"
    - 그 외  : 가장 가까운(20대)로 폴백
    """
    if age <= 19:
        return f"teen_{gender}", "10대"
    if 20 <= age <= 29:
        return f"twenties_{gender}", "20대"
    # 폴백 (현재 teen, twenties만 있다고 했으니 20대로 통일)
    return f"twenties_{gender}", "20대"

def temp_to_category(temp_c: float) -> str:
    """기온 → '더움'/'선선'/'추움' 구간화"""
    if temp_c >= 24:
        return "더움"
    if temp_c >= 15:
        return "선선"
    return "추움"

def get_weather(city_kor: str):
    """
    현재 기온/설명 가져오기. 실패시 (None, None) 반환
    """
    # 도시 영문화 (없으면 원문 그대로 시도)
    city_query = KOR_TO_ENG_CITY.get(city_kor.strip(), city_kor.strip())
    if not OPENWEATHER_API_KEY:
        return None, None

    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city_query}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr"
        )
        r = requests.get(url, timeout=8)
        data = r.json()
        if r.status_code == 200 and "main" in data:
            temp = float(data["main"]["temp"])
            desc = data["weather"][0]["description"]
            return temp, desc
    except Exception:
        pass
    return None, None

def safe_pick(d: dict, key: str):
    """
    d[key]가 없으면 가능한 첫 번째 값을 폴백으로 반환 (딕셔너리면 첫 키로 내려감)
    """
    if isinstance(d, dict):
        if key in d:
            return d[key]
        # 폴백: 첫 번째 항목
        if d:
            first_key = next(iter(d.keys()))
            return d[first_key]
    return None

def load_outfit(persona: str, gender_norm: str, age_label: str, weather_cat: str, skin: str, body: str):
    """
    JSON 구조: {persona: {gender: {age_label: {weather_cat: {skin: {body: {...}}}}}}}
    키가 없을 때는 단계적으로 폴백.
    """
    # 폴백 키
    default_skin = "웜톤"
    default_body = "보통"

    # 파일 경로 결정
    # 디렉터리는 뷰 함수에서 미리 계산해 넘겨주므로 여기선 전달된 age_label만 사용
    # 실제 파일은 persona.json (예: trendy.json)
    return {
        "resolver": lambda p: p  # 자리표시자 (아래에서 교체)
    }

def pick_from_json(json_path: Path, persona: str, gender_norm: str, age_label: str,
                   weather_cat: str, skin: str, body: str):
    """
    실제 JSON에서 탐색 + 폴백
    """
    default_skin = "웜톤"
    default_body = "보통"

    if not json_path.exists():
        return None

    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    # 단계별 안전 탐색
    node = safe_pick(data, persona)
    node = safe_pick(node, gender_norm)
    node = safe_pick(node, age_label)
    node = safe_pick(node, weather_cat)

    # 스킨 톤
    node_skin = safe_pick(node, skin)
    if node_skin is None:
        node_skin = safe_pick(node, default_skin)

    # 체형
    outfit = safe_pick(node_skin, body)
    if outfit is None:
        outfit = safe_pick(node_skin, default_body)

    # outfit 예: {"상의": "...", "하의": "...", "신발": "...", "액세서리": "..."}
    if isinstance(outfit, dict):
        return outfit
    return None

@app.route("/", methods=["GET"])
def select():
    return render_template(
        "chat_select.html",
        persona_label=PERSONA_LABEL,
        persona_image=PERSONA_IMAGE
    )

@app.route("/start", methods=["POST"])
def start():
    persona = request.form.get("persona", "").strip()
    if persona not in PERSONA_LABEL:
        return jsonify({"ok": False, "error": "캐릭터 선택이 올바르지 않습니다."}), 400

    session.clear()
    session["persona"] = persona
    session["step"] = 0
    session["answers"] = {}

    return jsonify({"ok": True})

@app.route("/chat", methods=["GET"])
def chat_page():
    persona = session.get("persona", "")
    label = PERSONA_LABEL.get(persona, persona)
    image = PERSONA_IMAGE.get(persona, "")
    return render_template("chat.html", persona=persona, persona_label=label, persona_image=image)

@app.route("/chat", methods=["POST"])
def chat_api():
    # 현재 단계
    step_idx = session.get("step", 0)
    answers = session.get("answers", {})
    persona = session.get("persona", "")

    user_msg = (request.json or {}).get("message", "").strip()

    # 단계 처리
    if step_idx == 0:  # 지역
        if not user_msg:
            return jsonify({"reply": "어느 지역에 계신가요? (예: 서울, 부산)"})
        answers["location"] = user_msg
        session["step"] = 1
        session["answers"] = answers
        return jsonify({"reply": "성별을 알려주세요. (남/여)"})

    if step_idx == 1:  # 성별
        g = normalize_gender(user_msg)
        if not g:
            return jsonify({"reply": "성별을 남/여 로 입력해주세요."})
        answers["gender"] = g
        session["step"] = 2
        session["answers"] = answers
        return jsonify({"reply": "나이를 숫자로 입력해주세요. (예: 17, 22)"})

    if step_idx == 2:  # 나이
        try:
            age = int(user_msg)
            if age <= 0 or age > 120:
                raise ValueError
        except Exception:
            return jsonify({"reply": "나이는 1~120 사이의 숫자로 입력해주세요."})
        answers["age"] = age
        session["step"] = 3
        session["answers"] = answers
        return jsonify({"reply": "피부 톤을 알려주세요. (웜톤/쿨톤)"})

    if step_idx == 3:  # 피부톤
        tone = user_msg.replace("톤", "").strip()
        if tone not in ["웜", "쿨", "웜톤", "쿨톤"]:
            return jsonify({"reply": "피부 톤은 웜톤/쿨톤 중에서 선택해주세요."})
        answers["skin_tone"] = "웜톤" if "웜" in tone else "쿨톤"
        session["step"] = 4
        session["answers"] = answers
        return jsonify({"reply": "체형을 알려주세요. (마른/보통/통통/역삼각형)"})

    if step_idx == 4:  # 체형 → 추천
        if user_msg not in ["마른", "보통", "통통", "역삼각형"]:
            return jsonify({"reply": "체형은 마른/보통/통통/역삼각형 중에서 입력해주세요."})
        answers["body_shape"] = user_msg

        # 날씨 조회
        temp, desc = get_weather(answers["location"])
        # 날씨 카테고리
        if temp is not None:
            weather_cat = temp_to_category(temp)
        else:
            # API 키 없음, 실패 등 폴백
            weather_cat = "선선"
            desc = "날씨 조회 실패(임시로 선선 적용)"

        # 나이/디렉터리/라벨
        age_dir, age_label = age_to_label_and_dir(int(answers["age"]), answers["gender"])

        # json 파일 경로
        json_file = JSON_ROOT / age_dir / f"{persona}.json"
        outfit = pick_from_json(
            json_file,
            persona=persona,
            gender_norm=answers["gender"],
            age_label=age_label,
            weather_cat=weather_cat,
            skin=answers["skin_tone"],
            body=answers["body_shape"]
        )

        if not outfit:
            reply = (
                f"선택 조합에 대한 코디 데이터를 찾지 못했습니다. "
                f"(캐릭터: {PERSONA_LABEL.get(persona, persona)}, 성별: {answers['gender']}, "
                f"나이: {answers['age']}, 날씨: {weather_cat}, 피부톤: {answers['skin_tone']}, 체형: {answers['body_shape']})\n"
                f"JSON 파일을 확인해주세요."
            )
        else:
            # 결과 문자열 구성
            # 도시 표기는 사용자가 입력한 그대로
            loc = answers["location"]
            weather_line = f"{loc}의 현재 날씨: {desc if desc else '정보 없음'}"
            outfit_line = (
                f"[오늘의 코디]\n"
                f"- 상의: {outfit.get('상의','')}\n"
                f"- 하의: {outfit.get('하의','')}\n"
                f"- 신발: {outfit.get('신발','')}\n"
                f"- 액세서리: {outfit.get('액세서리','')}"
            )
            reply = (
                f"{weather_line}\n"
                f"(분류: {weather_cat}, 나이대: {age_label}, 피부톤: {answers['skin_tone']}, 체형: {answers['body_shape']})\n\n"
                f"{outfit_line}"
            )

        # 대화 초기화 옵션 제공
        session["step"] = 0
        session["answers"] = {}
        return jsonify({"reply": reply})

    # 그 외 예외
    session["step"] = 0
    session["answers"] = {}
    return jsonify({"reply": "대화를 다시 시작합니다. 어느 지역에 계신가요? (예: 서울, 부산)"})


# Render 배포용 (개발용 서버도 지원)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
