from flask import Flask, render_template, request, session, jsonify
import requests
import os
import random

app = Flask(__name__)

# 보안용 시크릿키는 환경변수에서 가져옵니다. (Render에 등록)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key")

# OpenWeatherMap API 키 (환경변수에서)
API_KEY = os.environ.get("OPENWEATHER_API_KEY", "")

korean_to_english = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu",
    "인천": "Incheon", "광주": "Gwangju", "대전": "Daejeon",
    "울산": "Ulsan", "세종": "Sejong", "경기": "Gyeonggi-do",
    "강원": "Gangwon-do", "충북": "Chungcheongbuk-do", "충남": "Chungcheongnam-do",
    "전북": "Jeollabuk-do", "전남": "Jeollanam-do", "경북": "Gyeongsangbuk-do",
    "경남": "Gyeongsangnam-do", "제주": "Jeju"
}

def get_age_group(age):
    try:
        age = int(age)
    except:
        return None
    if age < 20:
        return "10대"
    if age < 30:
        return "20대"
    if age < 40:
        return "30대"
    return "40대 이상"

def build_outfit_list(temp, gender, age_group):
    # 각 조합마다 여러 추천을 넣어둡니다. 랜덤으로 최대 3개를 선택해 반환.
    suggestions = []

    gender = gender.strip()
    # normalize
    if gender.startswith("남"):
        gender_key = "남성"
    elif gender.startswith("여"):
        gender_key = "여성"
    else:
        gender_key = gender

    # 남성 추천 예시
    if gender_key == "남성":
        if age_group == "10대":
            if temp >= 27:
                suggestions = [
                    "그래픽 반팔티 + 카고 반바지 + 운동화 + 볼캡",
                    "루즈핏 나시 + 농구 반바지 + 샌들 + 체인 목걸이",
                    "오버핏 반팔 셔츠 + 데님 숏팬츠 + 스니커즈"
                ]
            elif temp >= 20:
                suggestions = [
                    "얇은 후드티 + 데님 팬츠 + 스니커즈",
                    "셔츠 + 면바지 + 캔버스화",
                    "라운드넥 티 + 치노팬츠 + 로퍼"
                ]
            elif temp >= 10:
                suggestions = [
                    "니트 + 조거 팬츠 + 코트 느낌의 아우터",
                    "맨투맨 + 청바지 + 운동화",
                    "가죽 재킷 + 슬림 데님 + 첼시 부츠"
                ]
            else:
                suggestions = [
                    "패딩 + 기모 후드 + 트레이닝 팬츠 + 방한 부츠",
                    "울 코트 + 터틀넥 + 기모 슬랙스 + 머플러"
                ]
        elif age_group == "20대":
            if temp >= 27:
                suggestions = [
                    "린넨 셔츠 + 반바지 + 로퍼",
                    "오버핏 셔츠 + 반바지 + 선글라스",
                    "슬림 탱크 + 코튼 쇼츠 + 샌들"
                ]
            elif temp >= 20:
                suggestions = [
                    "청자켓 + 반팔티 + 슬랙스 + 스니커즈",
                    "맨투맨 + 슬랙스 + 스니커즈",
                    "셔츠 + 치노팬츠 + 로퍼"
                ]
            elif temp >= 10:
                suggestions = [
                    "트렌치 코트 + 니트 + 청바지 + 첼시부츠",
                    "가디건 + 셔츠 + 슬랙스",
                    "울 재킷 + 데님 + 부츠"
                ]
            else:
                suggestions = [
                    "롱패딩 + 니트 + 기모 슬랙스 + 머플러",
                    "무스탕 + 후드 + 데님 + 방한화",
                ]
        else:
            # 30대 이상 공통 추천
            if temp >= 20:
                suggestions = [
                    "셔츠 + 슬랙스 + 로퍼",
                    "깔끔한 니트 + 치노 + 로퍼"
                ]
            else:
                suggestions = [
                    "울 코트 + 니트 + 슬랙스 + 부츠",
                    "패딩 + 터틀넥 + 방한화"
                ]

    # 여성 추천 예시
    elif gender_key == "여성":
        if age_group == "10대":
            if temp >= 27:
                suggestions = [
                    "크롭티 + 플리츠 스커트 + 운동화 + 볼캡",
                    "나시 + 와이드 팬츠 + 샌들",
                    "미니 원피스 + 샌들"
                ]
            elif temp >= 20:
                suggestions = [
                    "셔츠 + 반바지 + 스니커즈",
                    "긴팔 티 + 데님 스커트 + 캔버스화",
                    "블라우스 + 슬랙스 + 플랫슈즈"
                ]
            elif temp >= 10:
                suggestions = [
                    "후드 집업 + 조거팬츠 + 스니커즈",
                    "가디건 + 플레어 팬츠 + 플랫"
                ]
            else:
                suggestions = [
                    "롱패딩 + 기모 맨투맨 + 레깅스 + 롱부츠",
                    "롱코트 + 니트 + 울 팬츠 + 머플러"
                ]
        elif age_group == "20대":
            if temp >= 27:
                suggestions = [
                    "민소매 원피스 + 샌들 + 챙모자",
                    "린넨 셋업 + 플랫슈즈"
                ]
            elif temp >= 20:
                suggestions = [
                    "셔츠 원피스 + 스니커즈",
                    "블라우스 + 데님 + 로퍼"
                ]
            elif temp >= 10:
                suggestions = [
                    "트렌치코트 + 원피스 + 앵클부츠",
                    "니트 + 슬랙스 + 로퍼"
                ]
            else:
                suggestions = [
                    "롱코트 + 터틀넥 + 울 팬츠 + 부츠",
                    "숏패딩 + 기모 니트 + 팬츠 + 워커"
                ]
        else:
            # 30대 이상
            if temp >= 20:
                suggestions = [
                    "블라우스 + 세미 슬랙스 + 플랫슈즈",
                    "트렌치코트 + 원피스"
                ]
            else:
                suggestions = [
                    "울 코트 + 니트 + 슬랙스 + 부츠",
                    "패딩 + 터틀넥 + 방한화"
                ]
    else:
        # 성별을 모르거나 입력이 이상할 때 (중립 추천)
        if temp >= 20:
            suggestions = [
                "편안한 티셔츠 + 면바지 + 스니커즈",
                "셔츠 + 슬랙스 + 로퍼"
            ]
        else:
            suggestions = [
                "니트 + 코트 + 청바지 + 부츠",
                "패딩 + 기모바지 + 방한화"
            ]

    # 중복 제거 & 랜덤 샘플 (최대 3개)
    unique = list(dict.fromkeys(suggestions))
    take = min(3, len(unique))
    return random.sample(unique, take) if unique else ["기본적인 편안한 옷차림을 추천해요."]

@app.route("/")
def home():
    # 새 대화 시작
    session.clear()
    session["step"] = 1
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_msg = data.get("message", "").strip()
    step = session.get("step", 1)

    # STEP 1: 지역 입력
    if step == 1:
        session["city"] = user_msg
        session["step"] = 2
        return jsonify({"reply": "성별을 입력해주세요 (남성 / 여성)"})

    # STEP 2: 성별 입력
    if step == 2:
        # 간단 검증
        if not user_msg:
            return jsonify({"reply": "성별을 입력해 주세요 (예: 남성 또는 여성)."})
        session["gender"] = user_msg
        session["step"] = 3
        return jsonify({"reply": "나이를 숫자로 입력해주세요 (예: 25)"})

    # STEP 3: 나이 입력 -> 최종 결과
    if step == 3:
        if not user_msg.isdigit():
            return jsonify({"reply": "나이는 숫자만 입력해 주세요."})
        session["age"] = int(user_msg)
        session["step"] = 4

        # 날씨 호출 준비
        city_kor = session.get("city", "")
        city_eng = korean_to_english.get(city_kor)
        if not city_eng:
            return jsonify({"reply": f"죄송해요. '{city_kor}'는 현재 지원하지 않는 지역입니다. 다른 지역을 입력해주세요."})

        if not API_KEY:
            return jsonify({"reply": "서버에 OpenWeatherMap API 키가 설정되어 있지 않습니다. 관리자에게 문의하세요."})

        url = f"http://api.openweathermap.org/data/2.5/weather?q={city_eng}&appid={API_KEY}&units=metric&lang=kr"
        try:
            res = requests.get(url, timeout=8)
        except Exception as e:
            return jsonify({"reply": f"날씨 API 호출 중 오류가 발생했습니다: {e}"})

        if res.status_code != 200:
            return jsonify({"reply": "날씨 정보를 가져오지 못했습니다. API 키나 도시명을 확인하세요."})

        data = res.json()
        temp = data.get("main", {}).get("temp")
        desc = data.get("weather", [{}])[0].get("description", "")

        age_group = get_age_group(session["age"])
        outfits = build_outfit_list(temp, session["gender"], age_group)

        # 문자열로 합치기
        outfit_text = "\n".join([f"- {o}" for o in outfits])
        reply = f"{city_kor}의 현재 온도는 {temp}°C, 날씨는 '{desc}'입니다.\n{age_group} {session['gender']}을(를) 위한 추천 코디:\n{outfit_text}"
        return jsonify({"reply": reply})

    return jsonify({"reply": "대화를 다시 시작하려면 페이지를 새로고침하세요."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
