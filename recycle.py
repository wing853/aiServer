import os
# [중요] 최신 PyTorch 보안 로드 문제를 해결하기 위해 가장 먼저 설정해야 합니다.
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import tempfile
import jwt
from functools import wraps

app = Flask(__name__)

# 환경 설정
JWT_SECRET = os.environ.get("JWT_SECRET", "my-super-secure-and-long-secret-key-1234567890")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT = os.path.join(BASE_DIR, "best.pt")

# 모델 로드 (서버 기동 시 1회)
try:
    model = YOLO(WEIGHT)
    print("✅ 모델 로드 성공")
except Exception as e:
    print(f"❌ 모델 로드 실패: {e}")

# 분리 배출 가이드 데이터
DISPOSAL = {
    "캔":      "캔 전용 수거함에 버려주세요.",
    "플라스틱":  "플라스틱 전용 수거함에 버려주세요.",
    "종이":    "종이류 전용 수거함에 버려주세요.",
    "비닐":    "비닐 전용 수거함에 버려주세요.",
    "페트병":  "페트병 전용 수거함에 버려주세요.",
    "유리병":  "유리 전용 수거함에 버려주세요.",
    "스티로폼": "스티로폼 전용 수거함에 버려주세요."
}

# --- 라우팅 ---

@app.route("/", methods=["GET"])
def health_check():
    return "AI 서버가 정상적으로 실행 중입니다."

@app.route("/recycle/analyze", methods=["POST"])
def analyze_image():
    # 1. 파일 확인
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 필요합니다."}), 400

    image_file = request.files['image']
    
    # 2. 임시 파일을 생성하여 이미지 처리
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image_file.save(temp.name)
            temp_path = temp.name
        
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "이미지를 읽을 수 없습니다."}), 400

        # 3. 모델 추론
        results = model(img, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return jsonify({"error": "분석 결과가 없습니다."}), 400

        # 가장 신뢰도가 높은 결과 추출
        best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
        category = model.names[int(best_box.cls[0])]
        confidence = float(best_box.conf[0])
        
        # 4. 결과 반환
        return jsonify({
            "category": category,
            "confidence": round(confidence, 4),
            "disposal_method": DISPOSAL.get(category, "일반 쓰레기통에 버려주세요.")
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "서버 내부 오류"}), 500
    
    finally:
        # 임시 파일 삭제
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # Render 포트 바인딩 설정
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)