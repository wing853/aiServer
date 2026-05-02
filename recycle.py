import os
import torch
import warnings
import collections
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import tempfile

# 🔧 환경 설정
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
warnings.filterwarnings("ignore")

# 🔧 최소 safe globals (Pickle 관련 보안 에러 방지)
try:
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            collections.OrderedDict,
            torch.Size,
            torch._utils._rebuild_tensor_v2,
            torch.storage._load_from_bytes
        ])
except Exception as e:
    print(f"⚠️ 초기 설정 참고: {e}")

app = Flask(__name__)

# 📁 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_FIX = os.path.join(BASE_DIR, "best_fix.pt")
WEIGHT_ORIGIN = os.path.join(BASE_DIR, "best.pt")

model = None

# ✅ 모델 로드 전략 (단순화 및 안정화)
def load_model_strategy():
    global model
    try:
        # 1. 파일 경로 우선순위 설정
        weights_to_try = [WEIGHT_FIX, WEIGHT_ORIGIN, "yolov8n.pt"]
        
        for weight_path in weights_to_try:
            if weight_path == "yolov8n.pt" or os.path.exists(weight_path):
                print(f"🔄 모델 로드 시도 중: {weight_path}")
                # YOLO() 직접 호출이 가장 안전합니다 (구조 자동 매칭)
                model = YOLO(weight_path)
                print(f"✅ 모델 로드 성공: {weight_path}")
                break
        
        if model:
            model.to('cpu')  # Render 무료 티어 메모리 관리
            print("🚀 모델 CPU 모드 전환 완료")
        else:
            print("❌ 로드할 수 있는 모델 파일이 없습니다.")

    except Exception as e:
        print(f"❌ 모델 로드 중 최종 실패: {e}")
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

# 🩺 헬스 체크 및 통합 엔드포인트
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
        # 임시 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image_file.save(temp.name)
            temp_path = temp.name

        # 이미지 읽기
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "이미지 파일 해석 실패"}), 400

        # 메모리 최적화를 위한 리사이징
        img = cv2.resize(img, (640, 640))

        # 추론 실행
        results = current_model(img, verbose=False)

        # 결과 확인
        if not results or len(results[0].boxes) == 0:
            return jsonify({
                "category": "미인식",
                "confidence": 0.0,
                "disposal_method": "객체를 인식하지 못했습니다. 다시 촬영해주세요."
            })

        # 가장 신뢰도가 높은 결과 추출
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
        # 임시 파일 삭제 (메모리/용량 관리)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

if __name__ == "__main__":
    # Render 환경 대응 포트 설정
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)