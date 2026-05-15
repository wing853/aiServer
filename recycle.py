import os
import torch
import warnings
import collections
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import tempfile

app = Flask(__name__)

# 📁 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_FIX = os.path.join(BASE_DIR, "best_fix.pt")
WEIGHT_ORIGIN = os.path.join(BASE_DIR, "best.pt")

model = None

# ✅ 모델 로드
def load_model_strategy():
    global model
    weights_to_try = [WEIGHT_FIX, WEIGHT_ORIGIN]

    for weight_path in weights_to_try:
        if os.path.exists(weight_path):
            try:
                print(f"🔄 모델 로드 시도 중: {weight_path}")
                model = YOLO(weight_path)
                model.to('cpu')
                print(f"✅ 모델 로드 성공: {weight_path}")
                return
            except Exception as e:
                print(f"❌ {weight_path} 실패: {e}")
                continue

    print("❌ 모든 모델 로드 실패")
    model = None

def get_model():
    global model
    if model is None:
        load_model_strategy()
    return model

# ♻️ 분리배출 안내 데이터베이스
DISPOSAL = {
    "캔": "캔 전용 수거함에 버려주세요.",
    "플라스틱": "플라스틱 전용 수거함에 버려주세요.",
    "종이": "종이류 전용 수거함에 버려주세요.",
    "비닐": "비닐 전용 수거함에 버려주세요.",
    "페트병": "페트병 전용 수거함에 버려주세요.",
    "유리병": "유리 전용 수거함에 버려주세요.",
    "스티로폼": "스티로폼 전용 수거함에 버려주세요."
}

# 🩺 헬스 체크
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return analyze_image()
    return "✅ AI Server is Live (Render)"

# 🔍 이미지 분석 API
@app.route("/recycle/analyze", methods=["POST"])
def analyze_image():
    current_model = get_model()

    if current_model is None:
        return jsonify({"error": "모델 로드 실패 (서버 로그 확인)"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 전송되지 않았습니다 (key: image)"}), 400

    image_file = request.files['image']
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image_file.save(temp.name)
            temp_path = temp.name

        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "이미지 파일 해석 실패"}), 400

        img = cv2.resize(img, (640, 640))

        results = current_model(img, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return jsonify({
                "category": "미인식",
                "confidence": 0.0,
                "disposal_method": "객체를 인식하지 못했습니다. 다시 촬영해주세요."
            })

        best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
        class_id = int(best_box.cls[0])
        category = current_model.names[class_id]
        confidence = float(best_box.conf[0])

        return jsonify({
            "category": category,
            "confidence": round(confidence, 4),
            "disposal_method": DISPOSAL.get(category, "일반 쓰레기로 배출하거나 분리배출함을 확인하세요.")
        })

    except Exception as e:
        print(f"⚠️ 분석 중 에러 발생: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)