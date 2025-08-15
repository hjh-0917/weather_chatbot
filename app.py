from flask import Flask, render_template, request, jsonify, session
import requests
import os
import random

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

API_KEY = os.environ.get("OPENWEATHER_API_KEY")

# 한글 → 영어 도시명
korean_to_english = {
    "서울": "Seoul", "부산": "Busan", "대구": "Daegu",
    "인천": "Incheon", "광주": "Gwangju", "대전": "Daejeon",
    "울산": "Ulsan", "제주": "Jeju"
}

# 캐릭터 목록 (키 = 이미지 파일명/내부 키)
characters = {
    "trendy": "트렌디 전문가",
    "practical": "실속파 코디 장인",
    "luxury": "럭셔리 스타일리스트",
    "cute": "귀여운 패션 친구",
    "gentle": "신사 스타일러"
}

# 캐릭터별·성별·날씨(더움/선선/추움) 아이템 라이브러리
# 각 리스트에서 랜덤으로 뽑아 조합합니다.
ITEMS = {
    "trendy": {
        "남성": {
            "더움": {
                "top":    ["오버핏 그래픽 티", "메쉬 나시", "파스텔 스트라이프 셔츠"],
                "bottom": ["와이드 카고 쇼츠", "데님 반바지", "크롭 조거팬츠"],
                "shoes":  ["청키 스니커즈", "스포츠 샌들", "캔버스 스니커즈"],
                "outer":  [],
                "acc":    ["버킷햇", "실버 체인", "캡 모자"]
            },
            "선선": {
                "top":    ["오버핏 후드", "루즈핏 셔츠", "니트 폴로"],
                "bottom": ["와이드 데님", "카고 팬츠", "테크 조거팬츠"],
                "shoes":  ["레트로 스니커즈", "하이탑 스니커즈", "첼시 부츠"],
                "outer":  ["데님 자켓", "MA-1 점퍼", "코치 자켓"],
                "acc":    ["비니", "크로스백", "스포츠 캡"]
            },
            "추움": {
                "top":    ["헤비 후드", "터틀넥 니트", "플리스 풀오버"],
                "bottom": ["기모 조거팬츠", "블랙 슬림 진", "카고 팬츠"],
                "shoes":  ["청키 하이탑", "러너 스니커즈", "워커"],
                "outer":  ["숏패딩", "롱패딩", "보아 플리스 자켓"],
                "acc":    ["목도리", "비니", "백팩"]
            }
        },
        "여성": {
            "더움": {
                "top":    ["크롭티", "슬리브리스 탑", "시스루 셔츠"],
                "bottom": ["테니스 스커트", "데님 쇼츠", "와이드 팬츠"],
                "shoes":  ["청키 스니커즈", "플랫폼 샌들", "캔버스 스니커즈"],
                "outer":  [],
                "acc":    ["버킷햇", "스몰 숄더백", "체인 목걸이"]
            },
            "선선": {
                "top":    ["카라 니트", "루즈핏 스웨트셔츠", "크롭 가디건"],
                "bottom": ["하이웨스트 데님", "카고 스커트", "와이드 슬랙스"],
                "shoes":  ["레트로 스니커즈", "첼시 부츠", "로퍼"],
                "outer":  ["바시티 자켓", "숏 트렌치", "가죽 자켓"],
                "acc":    ["비니", "미니 크로스백", "헤어클립"]
            },
            "추움": {
                "top":    ["터틀넥 니트", "플리스 집업", "케이블 니트"],
                "bottom": ["기모 와이드 팬츠", "블랙 스키니", "롱 스커트"],
                "shoes":  ["앵클 부츠", "워커", "하이탑 스니커즈"],
                "outer":  ["숏패딩", "롱패딩", "울 코트"],
                "acc":    ["머플러", "비니", "토트백"]
            }
        }
    },
    "practical": {
        "남성": {
            "더움": {
                "top":    ["기능성 반팔 티", "린넨 셔츠", "쿨맥스 폴로"],
                "bottom": ["코튼 반바지", "라이트 치노", "스트레치 슬랙스"],
                "shoes":  ["통기성 스니커즈", "샌들", "로퍼"],
                "outer":  [],
                "acc":    ["캡 모자", "간단한 가죽 벨트", "심플 손목시계"]
            },
            "선선": {
                "top":    ["얇은 니트", "옥스포드 셔츠", "헨리넥 티"],
                "bottom": ["치노 팬츠", "스트레이트 데님", "슬랙스"],
                "shoes":  ["데일리 스니커즈", "로퍼", "데저트 부츠"],
                "outer":  ["라이트 윈드브레이커", "코튼 자켓", "가디건"],
                "acc":    ["캔버스 토트", "심플 머플러", "가죽 스트랩 시계"]
            },
            "추움": {
                "top":    ["울 니트", "기모 스웨트", "히트텍 폴로"],
                "bottom": ["기모 치노", "두께감 있는 데님", "울 슬랙스"],
                "shoes":  ["워커", "방한 스니커즈", "첼시 부츠"],
                "outer":  ["경량 패딩", "파카", "울 코트"],
                "acc":    ["니트 비니", "머플러", "글러브"]
            }
        },
        "여성": {
            "더움": {
                "top":    ["린넨 블라우스", "기능성 반팔", "슬리브리스 니트"],
                "bottom": ["통바지", "코튼 쇼츠", "A라인 스커트"],
                "shoes":  ["플랫 샌들", "통기성 스니커즈", "로퍼"],
                "outer":  [],
                "acc":    ["햇", "슬림 벨트", "작은 크로스백"]
            },
            "선선": {
                "top":    ["라이트 가디건", "셔츠", "니트 티"],
                "bottom": ["세미 와이드 팬츠", "일자 데님", "미디 스커트"],
                "shoes":  ["플랫슈즈", "로퍼", "스니커즈"],
                "outer":  ["코튼 트렌치", "숏 패딩 베스트", "니트 가디건"],
                "acc":    ["스카프", "토트백", "심플 목걸이"]
            },
            "추움": {
                "top":    ["케이블 니트", "기모 후디", "터틀넥"],
                "bottom": ["울 슬랙스", "기모 레깅스", "롱 스커트"],
                "shoes":  ["앵클 부츠", "워커", "방한 스니커즈"],
                "outer":  ["경량 롱패딩", "울 코트", "후드 파카"],
                "acc":    ["머플러", "비니", "가죽 장갑"]
            }
        }
    },
    "luxury": {
        "남성": {
            "더움": {
                "top":    ["실크 혼방 셔츠", "피케 폴로", "리넨 블렌드 니트"],
                "bottom": ["테일러드 쇼츠", "라이트 울 슬랙스", "린넨 팬츠"],
                "shoes":  ["스웨이드 로퍼", "레더 샌들", "미니멀 스니커즈"],
                "outer":  [],
                "acc":    ["레더 벨트", "썬글라스", "얇은 브레이슬릿"]
            },
            "선선": {
                "top":    ["캐시미어 니트", "파인 게이지 니트 폴로", "옥스포드 셔츠"],
                "bottom": ["테일러드 슬랙스", "크리즈 데님", "버진울 팬츠"],
                "shoes":  ["페니 로퍼", "더비 슈즈", "첼시 부츠"],
                "outer":  ["언스트럭처드 블레이저", "수에이드 자켓", "가죽 재킷"],
                "acc":    ["실크 스카프", "레더 카드월렛", "클래식 워치"]
            },
            "추움": {
                "top":    ["캐시미어 터틀넥", "메리노 니트", "울 셔츠"],
                "bottom": ["플란넬 슬랙스", "울 데님", "테일러드 팬츠"],
                "shoes":  ["레더 부츠", "하이엔드 스니커즈", "더비 슈즈"],
                "outer":  ["더블 브레스트 코트", "캐시미어 코트", "다운 재킷"],
                "acc":    ["가죽 장갑", "캐시미어 머플러", "페도라"]
            }
        },
        "여성": {
            "더움": {
                "top":    ["실크 블라우스", "캐시미어 슬리브리스", "리넨 셋업 탑"],
                "bottom": ["테일러드 쇼츠", "실크 스커트", "린넨 팬츠"],
                "shoes":  ["스트랩 샌들", "레더 로퍼", "미니멀 스니커즈"],
                "outer":  [],
                "acc":    ["미니 레더백", "선글라스", "진주 이어링"]
            },
            "선선": {
                "top":    ["파인 캐시미어 니트", "실크 셔츠", "울 혼방 탑"],
                "bottom": ["플리츠 스커트", "크리즈 데님", "테일러드 팬츠"],
                "shoes":  ["메리제인", "로퍼", "앵클 부츠"],
                "outer":  ["버진울 코트", "수에이드 재킷", "트위드 자켓"],
                "acc":    ["실크 스카프", "미니 숄더백", "클래식 워치"]
            },
            "추움": {
                "top":    ["캐시미어 터틀넥", "울 니트", "플리스 라이너 탑"],
                "bottom": ["울 슬랙스", "니트 스커트", "헤비 데님"],
                "shoes":  ["레더 부츠", "앵클 부츠", "하이엔드 스니커즈"],
                "outer":  ["핸드메이드 코트", "롱 다운", "무톤 코트"],
                "acc":    ["캐시미어 머플러", "가죽 장갑", "이어머프"]
            }
        }
    },
    "cute": {
        "남성": {
            "더움": {
                "top":    ["파스텔 티셔츠", "소프트 폴로", "라운드넥 니트 티"],
                "bottom": ["밴딩 쇼츠", "라이트 치노", "연청 반바지"],
                "shoes":  ["캔버스 스니커즈", "샌들", "러너"],
                "outer":  [],
                "acc":    ["볼캡", "캔버스 토트", "얇은 팔찌"]
            },
            "선선": {
                "top":    ["부클 가디건", "파스텔 스웨트셔츠", "니트 베스트+셔츠"],
                "bottom": ["세미와이드 데님", "코튼 팬츠", "밴딩 슬랙스"],
                "shoes":  ["로퍼", "스니커즈", "데저트 부츠"],
                "outer":  ["숏 트렌치", "코튼 재킷", "가디건"],
                "acc":    ["머플러", "미니 크로스백", "비니"]
            },
            "추움": {
                "top":    ["케이블 니트", "폴라 니트", "양기모 맨투맨"],
                "bottom": ["기모 조거", "울 팬츠", "일자 데님"],
                "shoes":  ["앵클 부츠", "스니커즈", "워커"],
                "outer":  ["더플 코트", "숏패딩", "울 코트"],
                "acc":    ["니트 비니", "머플러", "손난로 파우치"]
            }
        },
        "여성": {
            "더움": {
                "top":    ["프릴 블라우스", "리본 슬리브리스", "크롭 티"],
                "bottom": ["플레어 스커트", "데님 쇼츠", "밴딩 와이드팬츠"],
                "shoes":  ["메ary 제인 플랫", "샌들", "스니커즈"],
                "outer":  [],
                "acc":    ["헤어리본", "미니 숄더백", "펄 네클리스"]
            },
            "선선": {
                "top":    ["가디건+탑 셋업", "케이블 니트", "셔츠+니트 베스트"],
                "bottom": ["하이웨스트 데님", "플리츠 스커트", "세미와이드 팬츠"],
                "shoes":  ["플랫슈즈", "로퍼", "앵클 부츠"],
                "outer":  ["숏 코트", "트렌치", "퍼카라 자켓"],
                "acc":    ["베레모", "리본 스카프", "미니 크로스백"]
            },
            "추움": {
                "top":    ["터틀넥 니트", "모헤어 니트", "기모 후드"],
                "bottom": ["울 롱스커트", "기모 레깅스", "두꺼운 데님"],
                "shoes":  ["퍼 안감 부츠", "앵클 부츠", "스니커즈"],
                "outer":  ["더플 코트", "롱패딩", "울 코트"],
                "acc":    ["퍼 머플러", "귀도리", "니트 비니"]
            }
        }
    },
    "gentle": {
        "남성": {
            "더움": {
                "top":    ["린넨 셔츠", "피케 폴로", "얇은 옥스포드 셔츠"],
                "bottom": ["플리츠 치노", "라이트 슬랙스", "린넨 팬츠"],
                "shoes":  ["페니 로퍼", "미니멀 스니커즈", "레더 샌들"],
                "outer":  [],
                "acc":    ["가죽 벨트", "클래식 워치", "선글라스"]
            },
            "선선": {
                "top":    ["메리노 니트", "셔츠+니트 베스트", "파인게이지 니트 폴로"],
                "bottom": ["테이퍼드 슬랙스", "스트레이트 데님", "치노 팬츠"],
                "shoes":  ["더비 슈즈", "로퍼", "첼시 부츠"],
                "outer":  ["트렌치 코트", "언스트럭처드 블레이저", "수에이드 자켓"],
                "acc":    ["실크 타이(선택)", "캐시미어 스카프", "레더 카드지갑"]
            },
            "추움": {
                "top":    ["캐시미어 터틀넥", "울 니트", "두께감 있는 셔츠"],
                "bottom": ["플란넬 슬랙스", "네이비 울 팬츠", "진청 데님"],
                "shoes":  ["레더 부츠", "더비 슈즈", "첼시 부츠"],
                "outer":  ["체스터필드 코트", "더블 코트", "다운 코트"],
                "acc":    ["가죽 장갑", "울 머플러", "페도라(선택)"]
            }
        },
        "여성": {
            "더움": {
                "top":    ["실크 블라우스", "린넨 셔츠", "니트 탑"],
                "bottom": ["세미와이드 슬랙스", "린넨 스커트", "테일러드 쇼츠"],
                "shoes":  ["로퍼", "스트랩 샌들", "미들힐"],
                "outer":  [],
                "acc":    ["심플 펜던트", "가죽 벨트", "토트백"]
            },
            "선선": {
                "top":    ["파인 니트", "셔츠+가디건", "니트 폴로"],
                "bottom": ["플리츠 스커트", "크리즈 데님", "슬랙스"],
                "shoes":  ["로퍼", "메리제인", "앵클 부츠"],
                "outer":  ["트렌치 코트", "울 자켓", "라이트 코트"],
                "acc":    ["실크 스카프", "레더 숄더백", "워치"]
            },
            "추움": {
                "top":    ["캐시미어 터틀넥", "울 니트", "기모 블라우스"],
                "bottom": ["울 슬랙스", "니트 스커트", "헤비 데님"],
                "shoes":  ["앵클 부츠", "레더 부츠", "로퍼"],
                "outer":  ["핸드메이드 코트", "롱 다운", "울 코트"],
                "acc":    ["머플러", "가죽 장갑", "핸드백"]
            }
        }
    }
}

def weather_category(temp: float) -> str:
    # 카테고리: 더움(≥23), 선선(10~22.9), 추움(<10)
    if temp >= 23:
        return "더움"
    elif temp >= 10:
        return "선선"
    else:
        return "추움"

def pick(lst):
    return random.choice(lst) if lst else None

def build_outfit(character: str, gender: str, temp: float, age: int) -> str:
    # 연령에 따라 실루엣/격식 가중치(문장에 살짝 반영)
    if age < 20:
        tone = "캐주얼한 무드"
    elif age < 30:
        tone = "균형 잡힌 데일리"
    elif age < 40:
        tone = "세련된 실루엣"
    else:
        tone = "단정하고 품격 있는 무드"

    cat = weather_category(temp)
    lib = ITEMS.get(character, {}).get(gender, {}).get(cat)
    if not lib:
        return "코디 데이터를 찾지 못했습니다."

    top = pick(lib["top"])
    bottom = pick(lib["bottom"])
    shoes = pick(lib["shoes"])
    outer = pick(lib["outer"])  # 더움이면 None일 수 있음
    acc = pick(lib["acc"])

    parts = [f"상의: {top}", f"하의: {bottom}", f"신발: {shoes}"]
    if outer:
        parts.insert(2, f"아우터: {outer}")
    if acc:
        parts.append(f"액세서리: {acc}")

    return f"{tone}로 추천합니다 — " + ", ".join(parts)

def fetch_weather(city_kor: str):
    city_eng = korean_to_english.get(city_kor)
    if not city_eng:
        return None
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city_eng}&appid={API_KEY}&units=metric&lang=kr"
    res = requests.get(url)
    if res.status_code != 200:
        return None
    data = res.json()
    return {
        "temp": data["main"]["temp"],
        "desc": data["weather"][0]["description"]
    }

@app.route("/")
def select_character():
    return render_template("chat_select.html", characters=characters)

@app.route("/start_chat", methods=["POST"])
def start_chat():
    session.clear()
    session["character"] = request.form.get("character")
    session["step"] = "ask_city"
    return render_template("chat.html", character=characters[session["character"]])

@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message", "").strip()
    step = session.get("step", "ask_city")

    # 1) 지역 묻기
    if step == "ask_city":
        info = fetch_weather(user_msg)
        if not info:
            return jsonify({"reply": "지원하지 않는 지역이거나 날씨 정보를 불러오지 못했습니다. 예: 서울, 부산, 제주"})
        session["city_kor"] = user_msg
        session["temp"] = float(info["temp"])
        session["desc"] = info["desc"]
        session["step"] = "ask_gender"
        return jsonify({"reply": f"{user_msg}의 현재 기온은 {info['temp']}°C, 날씨는 '{info['desc']}'입니다. 성별을 입력해주세요 (남성/여성)."})

    # 2) 성별 묻기
    if step == "ask_gender":
        if user_msg not in ["남성", "여성"]:
            return jsonify({"reply": "성별은 '남성' 또는 '여성'으로 입력해주세요."})
        session["gender"] = user_msg
        session["step"] = "ask_age"
        return jsonify({"reply": "나이를 입력해주세요."})

    # 3) 나이 묻기 → 코디 생성
    if step == "ask_age":
        if not user_msg.isdigit():
            return jsonify({"reply": "나이는 숫자로 입력해주세요."})
        age = int(user_msg)
        session["age"] = age

        character = session["character"]
        gender = session["gender"]
        temp = session["temp"]

        outfit = build_outfit(character, gender, temp, age)
        session["step"] = "done"
        city = session["city_kor"]
        desc = session["desc"]
        return jsonify({"reply": f"{city}의 기온 {temp}°C, 날씨 '{desc}'.\n오늘의 추천 코디({characters[character]}): {outfit}"})

    # 완료 후
    if step == "done":
        return jsonify({"reply": "대화를 다시 시작하려면 새로고침 해주세요."})

    return jsonify({"reply": "오류가 발생했습니다. 새로고침 후 다시 시도해주세요."})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
