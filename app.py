from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import json
import os

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# 1️⃣ 스타일별 캐릭터 딕셔너리
CHARACTERS = {
    "trendy": "트렌디",
    "practical": "프랙티컬",
    "luxury": "럭셔리",
    "gentle": "젠틀",
    "cute": "큐트"
}

# 2️⃣ JSON 파일 기본 경로
BASE_PATH = "JSON"  # JSON/teen_male/trendy.json 등

def load_codi_data(age_group, gender, style):
    """
    연령대, 성별, 스타일에 맞는 JSON 데이터를 불러옵니다.
    """
    file_path = os.path.join(BASE_PATH, f"{age_group}_{gender}", f"{style}.json")
    if not os.path.exists(file_path):
        return {}
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

# 3️⃣ 캐릭터 선택 화면
@app.route("/")
def index():
    return render_template("chat_select.html", characters=CHARACTERS)

# 4️⃣ 캐릭터 선택 후 챗 화면으로 이동
@app.route("/start_chat", methods=["POST"])
def start_chat():
    style = request.form.get("character")  # chat_select.html에서 hidden input name="character"
    if style not in CHARACTERS:
        style = "trendy"  # 기본값
    session["style"] = style
    return redirect(url_for("chat"))

# 5️⃣ 챗 화면
@app.route("/chat")
def chat():
    style = session.get("style", "trendy")
    return render_template("chat.html", character=CHARACTERS.get(style, "트렌디"))

# 6️⃣ 챗 메시지 처리
@app.route("/chat", methods=["POST"])
def chat_post():
    user_message = request.json.get("message")
    style = session.get("style", "trendy")
    
    # ⚠️ 실제 코디 추천 로직: JSON 데이터 불러와 연령대, 성별, 체형, 톤 등 활용 가능
    # 현재 예시는 단순 메시지 응답
    reply = f"{CHARACTERS.get(style)} 스타일 챗봇이 답변합니다: '{user_message}'"
    
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
