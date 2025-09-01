import os
import json
import csv

# JSON 폴더 경로
json_root = "json"

# 출력 CSV 파일
output_csv = "outfits.csv"

# CSV에 쓸 필드
fieldnames = ["age_group", "gender", "persona", "weather", "skin_tone", "body_type", "top", "bottom"]

rows = []

# JSON 루프
for age_group in os.listdir(json_root):
    age_path = os.path.join(json_root, age_group)
    if not os.path.isdir(age_path):
        continue

    for gender in os.listdir(age_path):
        gender_path = os.path.join(age_path, gender)
        if not os.path.isdir(gender_path):
            continue

        for persona_file in os.listdir(gender_path):
            if not persona_file.endswith(".json"):
                continue

            persona_name = persona_file.replace(".json", "")
            json_file = os.path.join(gender_path, persona_file)

            with open(json_file, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"JSON decode error in {json_file}")
                    continue

            # data 안에 outfits 리스트가 있다고 가정
            for outfit in data.get("outfits", []):
                # outfit이 딕셔너리면 get 사용, 문자열이면 '+' 기준으로 분리
                if isinstance(outfit, dict):
                    top = outfit.get("상의", "")
                    bottom = outfit.get("하의", "")
                    weather = outfit.get("날씨", "")
                    skin_tone = outfit.get("피부톤", "")
                    body_type = outfit.get("체형", "")
                else:
                    parts = outfit.split("+")
                    top = parts[0].strip() if len(parts) > 0 else outfit
                    bottom = parts[1].strip() if len(parts) > 1 else ""
                    # JSON에 날씨, 피부톤, 체형 정보가 없으면 빈 문자열
                    weather = ""
                    skin_tone = ""
                    body_type = ""

                row = {
                    "age_group": age_group,
                    "gender": gender,
                    "persona": persona_name,
                    "weather": weather,
                    "skin_tone": skin_tone,
                    "body_type": body_type,
                    "top": top,
                    "bottom": bottom
                }
                rows.append(row)

# CSV 저장
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"CSV 파일 생성 완료: {output_csv}")


