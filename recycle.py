import os
import torch

# [필수] PyTorch 2.6+ 보안 로드 이슈 해결을 위한 안전 클래스 등록
# 모델 로드 시 발생하는 UnpicklingError를 근본적으로 차단합니다.
try:
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF
    from ultralytics.nn.modules.head import Detect

    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            DetectionModel, Conv, C2f, Bottleneck, SPPF, Detect,
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU
        ])
    # 환경 변수도 가장 확실한 시점에 다시 설정
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
except Exception as e:
    print(f"⚠️ 보안 설정 적용 중 알림: {e}")

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import tempfile

app = Flask(__name__)

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT = os.path.join(BASE_DIR, "best.pt")

# 모델 로드 (에러 발생 시 강제 로드 시도)
try:
    # 1차 시도
    model = YOLO(WEIGHT)
    print("✅✅ 모델 로드 대성공!")
except Exception as e:
    print(f"⚠️ 1차 로드 실패 후 재시도: {e}")
    try:
        # 2차 시도: 환경 변수를 강제로 적용하여 로드
        model = YOLO(WEIGHT) 
        print("✅ 모델 로드 성공 (재시도)")
    except Exception as e2:
        print(f"❌ 모델 로드 최종 실패: {e2}")
        model = None

DISPOSAL = {
    "캔": "캔 전용 수거함에 버려주세요.",
    "플라스틱": "플라스틱 전용 수거함에 버려주세요.",
    "종이": "종이류 전용 수거함에 버려주세요.",
    "비닐": "비닐 전용 수거함에 버려주세요.",
    "페트병": "페트병 전용 수거함에 버려주세요.",
    "유리병": "유리 전용 수거함에 버려주세요.",
    "스티로폼": "스티로폼 전용 수거함에 버려주세요."
}

@app.route("/", methods=["GET"])
def health():
    status = "정상" if model else "모델 미로드"
    return f"AI 서버 상태: {status}"

@app.route("/recycle/analyze", methods=["POST"])
def analyze_image():
    if model is None:
        return jsonify({"error": "모델이 로드되지 않아 분석할 수 없습니다."}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "이미지가 없습니다."}), 400

    image_file = request.files['image']
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image_file.save(temp.name)
            temp_path = temp.name
        
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "이미지 읽기 실패"}), 400

        results = model(img, verbose=False)
        
        if not results or len(results[0].boxes) == 0:
            return jsonify({"error": "감지된 객체 없음"}), 400

        best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
        category = model.names[int(best.cls[0])]
        
        return jsonify({
            "category": category,
            "confidence": round(float(best.conf[0]), 4),
            "disposal_method": DISPOSAL.get(category, "일반 쓰레기로 배출하세요.")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)