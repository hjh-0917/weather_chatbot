from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# ---------------------------------------
# 캐릭터 목록
# ---------------------------------------
CHARACTERS = {
    "trendy": "트렌디",
    "practical": "프랙티컬",
    "luxury": "럭셔리",
    "gentle": "젠틀",
    "cute": "큐트"
}

# ---------------------------------------
# 연령/성별별 JSON 파일 경로
# 예: json_data/teen_male/trendy.json
# ---------------------------------------
JSON_FOLDER = "json_data"

# ---------------------------------------
# 사용자 세션 데이터 저장 (간단히 전역 딕셔너리 사용)
# ---------------------------------------
USER_DATA = {}

# ---------------------------------------
# index: 캐릭터 선택 화면
# ---------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("chat_select.html", characters=CHARACTERS)

# ---------------------------------------
# start_chat: 캐릭터 선택 후 챗 시작
# ---------------------------------------
@app.route("/start_chat", methods=["POST"])
def start_chat():
    character = request.form.get("character")
    if not character or character not in CHARACTERS:
        return "잘못된 캐릭터 선택", 400
    
    # 사용자 세션 초기화
    user_id = request.remote_addr
    USER_DATA[user_id] = {
        "character": character,
        "stage": "지역"  # 챗봇 진행 단계
    }
    
    return render_template("chat.html", character=CHARACTERS[character])

# ---------------------------------------
# chat: 챗 메시지 처리
# ---------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    user_id = request.remote_addr
    user_session = USER_DATA.get(user_id)
    if not user_session:
        return jsonify({"reply": "세션이 없습니다. 캐릭터를 선택해주세요."})
    
    data = request.get_json()
    msg = data.get("message", "").strip()
    
    if not msg:
        return jsonify({"reply": "메시지를 입력해주세요."})
    
    stage = user_session["stage"]
    
    # 단계별 처리
    if stage == "지역":
        user_session["region"] = msg
        user_session["stage"] = "성별"
        reply = "좋아요. 성별을 입력해주세요. (남자/여자)"
    
    elif stage == "성별":
        if msg not in ["남자", "여자"]:
            reply = "성별은 '남자' 또는 '여자'로 입력해주세요."
        else:
            user_session["gender"] = msg
            user_session["stage"] = "연령"
            reply = "연령대를 입력해주세요. (10대/20대)"
    
    elif stage == "연령":
        if msg not in ["10대", "20대"]:
            reply = "연령대는 '10대' 또는 '20대'로 입력해주세요."
        else:
            user_session["age"] = msg
            user_session["stage"] = "체형"
            reply = "체형을 입력해주세요. (마른/보통/통통/역삼각형)"
    
    elif stage == "체형":
        if msg not in ["마른", "보통", "통통", "역삼각형"]:
            reply = "체형은 '마른/보통/통통/역삼각형' 중에서 입력해주세요."
        else:
            user_session["body"] = msg
            user_session["stage"] = "피부톤"
            reply = "피부톤을 입력해주세요. (웜톤/쿨톤)"
    
    elif stage == "피부톤":
        if msg not in ["웜톤", "쿨톤"]:
            reply = "피부톤은 '웜톤/쿨톤'으로 입력해주세요."
        else:
            user_session["tone"] = msg
            user_session["stage"] = "날씨"
            reply = "마지막으로 날씨를 입력해주세요. (더움/선선/추움)"
    
    elif stage == "날씨":
        if msg not in ["더움", "선선", "추움"]:
            reply = "날씨는 '더움/선선/추움' 중에서 입력해주세요."
        else:
            user_session["weather"] = msg
            # 추천 코디 가져오기
            reply = get_coordination(user_session)
    
    else:
        reply = "알 수 없는 단계입니다. 처음부터 다시 선택해주세요."
    
    return jsonify({"reply": reply})

# ---------------------------------------
# 추천 코디 가져오기 함수
# ---------------------------------------
def get_coordination(session):
    character = session["character"]
    gender = session["gender"]
    age = session["age"]
    body = session["body"]
    tone = session["tone"]
    weather = session["weather"]
    
    # JSON 파일 경로
    json_path = os.path.join(JSON_FOLDER, f"{age}_{gender.lower()}", character + ".json")
    
    if not os.path.exists(json_path):
        return f"{CHARACTERS[character]} 캐릭터의 코디 파일이 존재하지 않습니다."
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 딕셔너리 구조에 맞춰 코디 가져오기
    try:
        outfit = data[character][gender][age][weather][tone][body]
        reply = (
            f"추천 코디입니다:\n"
            f"상의: {outfit['상의']}\n"
            f"하의: {outfit['하의']}\n"
            f"신발: {outfit['신발']}\n"
            f"액세서리: {outfit['액세서리']}"
        )
        return reply
    except KeyError:
        return "해당 조건의 코디를 찾을 수 없습니다."

# ---------------------------------------
# 앱 실행
# ---------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

