import os
import torch
import warnings
import collections

# 🔧 환경 설정
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
warnings.filterwarnings("ignore")

# 🔧 최소 safe globals
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

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import tempfile

app = Flask(__name__)

# 📁 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_FIX = os.path.join(BASE_DIR, "best_fix.pt")
WEIGHT_ORIGIN = os.path.join(BASE_DIR, "best.pt")
BASE_MODEL = os.path.join(BASE_DIR, "yolov8n.pt")  # 🔥 직접 넣어야 함

model = None

# ✅ 모델 로드 (lazy loading)
def load_model_strategy():
    global model
    try:
        if os.path.exists(WEIGHT_FIX):
            print("🔄 best_fix.pt 사용 (가중치 주입)")

            model = YOLO(BASE_MODEL)

            state_dict = torch.load(WEIGHT_FIX, map_location='cpu')

            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model'].state_dict() \
                    if hasattr(state_dict['model'], 'state_dict') else state_dict['model']

            model.model.load_state_dict(state_dict, strict=False)

            print("✅ 가중치 주입 완료")

        elif os.path.exists(WEIGHT_ORIGIN):
            print("🔄 best.pt 직접 로드")
            model = YOLO(WEIGHT_ORIGIN)
            print("✅ 모델 로드 완료")

        else:
            print("❌ 모델 파일 없음")

        # 🔥 메모리 최적화
        if model:
            model.to('cpu')

    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        model = None


def get_model():
    global model
    if model is None:
        load_model_strategy()
    return model


# ♻️ 분리배출 안내
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
@app.route("/", methods=["GET"])
def health():
    return "✅ Server is Live"


# 🔍 이미지 분석
@app.route("/recycle/analyze", methods=["POST"])
def analyze_image():
    model = get_model()

    if model is None:
        return jsonify({"error": "모델 로드 실패"}), 500

    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일 없음"}), 400

    image_file = request.files['image']
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image_file.save(temp.name)
            temp_path = temp.name

        img = cv2.imread(temp_path)

        if img is None:
            return jsonify({"error": "이미지 읽기 실패"}), 400

        # 🔥 메모리 안정화 (중요)
        img = cv2.resize(img, (640, 640))

        results = model(img, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return jsonify({"error": "감지된 객체 없음"}), 400

        best = max(results[0].boxes, key=lambda b: float(b.conf[0]))

        category = model.names[int(best.cls[0])]
        confidence = float(best.conf[0])

        return jsonify({
            "category": category,
            "confidence": round(confidence, 4),
            "disposal_method": DISPOSAL.get(category, "일반 쓰레기")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)