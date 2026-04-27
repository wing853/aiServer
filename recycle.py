import os
import torch

# [1단계] PyTorch 2.6+ 보안 정책(WeightsUnpickler) 대응 설정
# 이 부분은 모든 import 중 가장 최상단에 위치해야 합니다.
try:
    import ultralytics
    # 에러 로그에서 지목한 모든 클래스를 '안전한 클래스'로 등록합니다.
    if hasattr(torch.serialization, 'add_safe_globals'):
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.nn.modules.conv import Conv
        from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF
        from ultralytics.nn.modules.head import Detect
        
        torch.serialization.add_safe_globals([
            DetectionModel, Conv, C2f, Bottleneck, SPPF, Detect,
            torch.nn.modules.pooling.MaxPool2d,      # <--- 방금 에러의 범인!
            torch.nn.modules.container.Sequential,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.conv.Conv2d,
            torch.nn.modules.batchnorm.BatchNorm2d,
            torch.nn.modules.activation.SiLU,
            torch.nn.modules.activation.LeakyReLU,
            torch.nn.modules.upsampling.Upsample,
            torch.storage._load_from_bytes,
            torch._utils._rebuild_tensor_v2,
            torch.Size
        ])
    
    # 이중 안전장치: 환경 변수를 통해 가중치 전용 로드 강제 해제
    os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
except Exception as e:
    print(f"⚠️ 초기 보안 설정 중 참고: {e}")

from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import tempfile

app = Flask(__name__)

# 경로 및 모델 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT = os.path.join(BASE_DIR, "best.pt")

# [2단계] 모델 로드 실행
model = None
try:
    # 위에서 설정한 add_safe_globals 덕분에 이제 정상 로드됩니다.
    model = YOLO(WEIGHT)
    print("✅ 모델 로드 성공!")
except Exception as e:
    print(f"❌ 모델 로드 최종 실패: {e}")

# 분리 배출 데이터 가이드
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
    if model:
        return "✅ AI Server is Live and Model is Loaded!"
    else:
        return "⚠️ AI Server is Live but Model Load Failed."

@app.route("/recycle/analyze", methods=["POST"])
def analyze_image():
    if model is None:
        return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
    
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    image_file = request.files['image']
    temp_path = None

    try:
        # 1. 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            image_file.save(temp.name)
            temp_path = temp.name
        
        # 2. 이미지 읽기 및 분석
        img = cv2.imread(temp_path)
        if img is None:
            return jsonify({"error": "이미지 읽기 실패"}), 400

        results = model(img, verbose=False)
        
        # 3. 결과 처리
        if not results or len(results[0].boxes) == 0:
            return jsonify({"error": "감지된 쓰레기 객체가 없습니다."}), 400

        # 가장 확률(Confidence)이 높은 결과 추출
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
        # 4. 임시 파일 삭제
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    # Render 환경의 PORT 대응
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)